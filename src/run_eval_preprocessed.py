# run_eval_preprocessed.py
# - 입력: preprocessed_data.csv (이미 전처리 완료)
# - 사용 컬럼: author_id(유저), app_id(아이템), title(제목), id(시간 대용), is_positive_encoded(0/1)
# - 모델: Hybrid = alpha*CBF(제목 TF-IDF) + (1-alpha)*Popularity
# - 출력: eval_outputs/per_user_metrics.csv, summary_metrics.csv, split_info.json

import os, json
from typing import List, Tuple, Iterable, Optional
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp  # 희소행렬 판별/변환용

# ================= 설정 ==================
CSV_PATH   = "preprocessed_data.csv"
USER_COL   = "author_id"
ITEM_COL   = "app_id"
TITLE_COL  = "title"
TIME_COL   = "id"                   # 시간 대용 (없으면 자동 생성)
TARGET_COL = "is_positive_encoded"  # 전처리에서 만든 0/1 라벨
HOLDOUT    = 1                      # 사용자별 마지막 N개를 테스트로
K_LIST     = [5, 10, 20]
ALPHA      = 0.7                    # 콘텐츠(CBF) 가중치
DEDUP_ONCE = True                   # 유저-아이템 중복 1건으로 축약(라벨=max, 시간=최신 id)
OUT_DIR    = "eval_outputs"
# ==========================================

# ---------- Metrics ----------
def precision_at_k(recs: List[str], gt: set, k: int) -> float:
    if k <= 0 or not recs: return 0.0
    topk = recs[:k]
    hits = sum(1 for x in topk if x in gt)
    return hits / k

def recall_at_k(recs: List[str], gt: set, k: int) -> float:
    if not gt or not recs: return 0.0
    topk = recs[:k]
    hits = sum(1 for x in topk if x in gt)
    return hits / len(gt)

def f1_at_k(recs: List[str], gt: set, k: int) -> float:
    p = precision_at_k(recs, gt, k)
    r = recall_at_k(recs, gt, k)
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))

# ---------- Split ----------
def chrono_holdout(df: pd.DataFrame, user_col: str, time_col: str, holdout: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_rows, test_rows = [], []
    for _, g in df.sort_values(time_col).groupby(user_col):
        if len(g) <= holdout:
            test_rows.append(g)
        else:
            test_rows.append(g.tail(holdout))
            train_rows.append(g.iloc[:-holdout])
    test_df = pd.concat(test_rows).reset_index(drop=True)
    train_df = pd.concat(train_rows).reset_index(drop=True) if train_rows else df.iloc[0:0].copy()
    return train_df, test_df

# ---------- Scorers ----------
class PopularityScorer:
    def __init__(self, interactions: pd.DataFrame, item_col: str, target_col: Optional[str]):
        if target_col and target_col in interactions.columns:
            pop = interactions.groupby(item_col)[target_col].sum()
        else:
            pop = interactions.groupby(item_col)[item_col].count()
        self.pop_scores = pop.astype(float)
        m = self.pop_scores.max()
        if m > 0: self.pop_scores /= m
    def score_all(self) -> pd.Series:
        return self.pop_scores.copy()

class CBFTitleScorer:
    """제목 TF-IDF 기반 콘텐츠 스코어러: 사용자 positive 아이템들과의 평균 코사인 유사도"""
    def __init__(self, items_df: pd.DataFrame, item_col: str, title_col: str):
        self.item_col = item_col
        self.title_col = title_col
        self.items = (items_df[[item_col, title_col]]
                      .drop_duplicates(subset=[item_col])
                      .fillna({title_col: ""})
                      .reset_index(drop=True))
        self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
        self.tfidf = self.vectorizer.fit_transform(self.items[title_col].astype(str).values)
        self.item_index = {it: idx for idx, it in enumerate(self.items[item_col].tolist())}

    def score_user(self, positives: Iterable[str]) -> pd.Series:
        pos_idx = [self.item_index[i] for i in positives if i in self.item_index]
        if not pos_idx:
            return pd.Series(0.0, index=self.items[self.item_col].tolist())

        # 평균 벡터 계산 (여기서 np.matrix가 생길 수 있음)
        user_mat = self.tfidf[pos_idx]
        user_profile = user_mat.mean(axis=0)  # shape: (1, n_features), 종종 numpy.matrix

        # ----- 핵심 수정: ndarray로 변환 -----
        if sp.issparse(user_profile):
            user_profile = user_profile.A  # to dense ndarray
        else:
            user_profile = np.asarray(user_profile)

        # cosine_similarity는 (n_samples, n_features) 형식을 기대
        # user_profile가 (n_features,)로 납작해지지 않도록 보장
        if user_profile.ndim == 1:
            user_profile = user_profile[np.newaxis, :]

        sims = cosine_similarity(user_profile, self.tfidf).ravel()
        return pd.Series(sims, index=self.items[self.item_col].tolist())

class HybridRecommender:
    def __init__(self, interactions_train: pd.DataFrame, items_df: pd.DataFrame,
                 user_col=USER_COL, item_col=ITEM_COL, title_col=TITLE_COL,
                 target_col: Optional[str]=TARGET_COL, alpha: float=ALPHA):
        self.user_col, self.item_col, self.title_col, self.target_col, self.alpha = user_col, item_col, title_col, target_col, alpha
        self.pop = PopularityScorer(interactions_train, item_col=item_col, target_col=target_col).score_all()
        self.cbf = CBFTitleScorer(items_df, item_col=item_col, title_col=title_col)
        self.user_pos = defaultdict(set)
        pos_df = interactions_train[interactions_train[target_col] > 0] if (target_col and target_col in interactions_train.columns) else interactions_train
        for u, g in pos_df.groupby(user_col):
            self.user_pos[u] = set(g[item_col].tolist())

    def recommend(self, user: str, topn: int = 100, exclude_seen: bool = True) -> list:
        cbf = self.cbf.score_user(self.user_pos.get(user, set()))
        pop = self.pop.reindex(cbf.index).fillna(0.0)
        scores = self.alpha * cbf + (1 - self.alpha) * pop  # <-- self.alpha 사용
        if exclude_seen and user in self.user_pos:
            scores = scores.drop(labels=list(self.user_pos[user] & set(scores.index)), errors="ignore")
        return scores.sort_values(ascending=False).head(topn).index.tolist()

# ---------- Evaluation ----------
def evaluate_topk(items_df, train_df, test_df, user_col, item_col, title_col, target_col, k_list, alpha):
    rec = HybridRecommender(train_df, items_df, user_col, item_col, title_col, target_col, alpha)
    rows = []
    for u, g in test_df.groupby(user_col):
        gt = set(g[item_col].tolist())
        if not gt: continue
        preds = rec.recommend(user=u, topn=max(k_list) + 50)
        row = {user_col: u, "gt_count": len(gt)}
        for k in k_list:
            row[f"P@{k}"] = precision_at_k(preds, gt, k)
            row[f"R@{k}"] = recall_at_k(preds, gt, k)
            row[f"F1@{k}"] = f1_at_k(preds, gt, k)
        rows.append(row)
    per_user = pd.DataFrame(rows).sort_values(user_col).reset_index(drop=True)
    summary = per_user[[c for c in per_user.columns if any(t in c for t in ["@5","@10","@20"])]].mean().to_frame("mean").T if len(per_user) else pd.DataFrame()
    return per_user, summary

# ---------- Main (자동 실행) ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 타입 통일(권장): 문자열 키 불일치 방지
    for c in [USER_COL, ITEM_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if TIME_COL not in df.columns:
        df[TIME_COL] = np.arange(len(df))

    need = {USER_COL, ITEM_COL, TITLE_COL, TIME_COL}
    if TARGET_COL in df.columns: need.add(TARGET_COL)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[list(need)].copy()

    # 중복 축약(권장): 라벨=max(1 있으면 1), 시간=최신 id
    if DEDUP_ONCE:
        agg = {TIME_COL: "max"}
        if TARGET_COL in data.columns: agg[TARGET_COL] = "max"
        data = data.groupby([USER_COL, ITEM_COL, TITLE_COL], as_index=False).agg(agg)

    train_df, test_df = chrono_holdout(data, USER_COL, TIME_COL, HOLDOUT)
    items_df = data[[ITEM_COL, TITLE_COL]].drop_duplicates(ITEM_COL)

    per_user, summary = evaluate_topk(items_df, train_df, test_df, USER_COL, ITEM_COL, TITLE_COL,
                                      TARGET_COL if TARGET_COL in data.columns else None, K_LIST, ALPHA)

    per_user_path = os.path.join(OUT_DIR, "per_user_metrics.csv")
    summary_path  = os.path.join(OUT_DIR, "summary_metrics.csv")
    split_info_path = os.path.join(OUT_DIR, "split_info.json")

    per_user.to_csv(per_user_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump({
            "csv": os.path.abspath(CSV_PATH),
            "user_col": USER_COL, "item_col": ITEM_COL,
            "title_col": TITLE_COL, "time_col": TIME_COL,
            "target_col": TARGET_COL if TARGET_COL in data.columns else None,
            "holdout": HOLDOUT, "k_list": K_LIST, "alpha": ALPHA,
            "dedup_once": bool(DEDUP_ONCE),
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Summary (macro-avg) ===")
    print(summary if len(summary) else "No summary (no test users).")
    print(f"\nSaved per-user metrics: {per_user_path}")
    print(f"Saved summary metrics : {summary_path}")
    print(f"Saved split info      : {split_info_path}")

if __name__ == "__main__":
    main()
