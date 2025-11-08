import argparse, os, random
import pandas as pd
from adapters import CBF_TFIDF, MF_Simple, HybridWeighted
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    # 핵심: dtype 강제로 과학적 표기/정밀도 손실 방지
    df = pd.read_csv(path, dtype={"author_id": str, "app_id": str})
    needed = ["author_id", "app_id", "title", "is_positive_encoded"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    # id 없으면 index로 생성 (문자열 유지)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    df["id"] = df["id"].astype(str)
    return df

def per_user_temporal_holdout(df: pd.DataFrame):
    # 같은 df에서 split하므로 dtype만 일치하면 키가 유지됩니다.
    df = df.sort_values(["author_id", "id"]).copy()
    train_idx, test_idx = [], []
    for uid, g in df.groupby("author_id", sort=False):
        if len(g) == 1:
            test_idx.extend(g.index.tolist())
        else:
            test_idx.append(g.index.tolist()[-1])
            train_idx.extend(g.index.tolist()[:-1])
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed_data.csv")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    df = load_data(args.data)
    train, test = per_user_temporal_holdout(df)

    item_titles = df.drop_duplicates(subset=["app_id"]).set_index("app_id")["title"]
    all_items = set(df["app_id"].unique().tolist())
    interacted_train = train.groupby("author_id")["app_id"].apply(set).to_dict()

    cbf = CBF_TFIDF(); cbf.fit(train, item_titles)
    cf  = MF_Simple();  cf.fit(train, item_titles)
    hyb = HybridWeighted(cf, cbf, alpha=args.alpha)

    # ✨ 교차 확인: train/test 사용자 키가 얼마나 겹치는지
    train_users = set(train["author_id"].unique())
    test_users  = set(test["author_id"].unique())
    inter_users = len(train_users & test_users)
    print(f"[check] #users train={len(train_users)}, test={len(test_users)}, intersect={inter_users}")

    rows = []
    for uid in sorted(test["author_id"].unique()):
        cands = list(all_items - interacted_train.get(uid, set()))
        for model_name, model in [("CBF", cbf), ("CF", cf), (f"Hybrid(a={args.alpha})", hyb)]:
            recs = model.recommend(uid, cands, topk=args.k)
            for rank, (app, score) in enumerate(recs, start=1):
                rows.append({"model": model_name, "author_id": uid, "app_id": app, "rank": rank, "score": score})

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    pos = test[test["is_positive_encoded"] == 1][["author_id", "app_id"]].copy()
    pos.to_csv("test_positive.csv", index=False)
    print(f"\n✅ Saved predictions to: {args.out}")
    print("✅ Saved test positives to: test_positive.csv")

    # ===== 중간 출력 & 진단 =====
    sample_users = random.sample(sorted(test["author_id"].unique().tolist()), k=min(3, len(test)))
    print("\n===== 샘플 추천 결과 (Top-5 + 진단) =====")
    for su in sample_users:
        n_inter = len(interacted_train.get(su, set()))
        print(f"\nUser: {su} | train_interactions={n_inter}")
        cands = out_df[out_df["author_id"] == su]["app_id"].unique().tolist()

        def top5(model_key):
            return out_df[(out_df["model"] == model_key) & (out_df["author_id"] == su)] \
                    .sort_values("rank").head(5)["app_id"].tolist()

        for m in ["CBF", "CF", f"Hybrid(a={args.alpha})"]:
            print(f"  {m:<15} → {', '.join(top5(m)) or '(no recs)'}")

        # (선택) 점수 분산 체크: 모두 동일점수면 경고
        for m, mdl in [("CBF", cbf), ("CF", cf), (f"Hybrid(a={args.alpha})", hyb)]:
            try:
                # adapters에 score_series(uid, cands) 가 있다면…
                # 없으면 이 블록은 무시됩니다.
                # 점수 분산이 0이면 사실상 '동일 점수 → 동일 랭킹' 상황
                s = mdl.score_series(su, cands) if hasattr(mdl, "score_series") else None
                if s is not None and len(s) > 0:
                    var = float(np.var(pd.to_numeric(s, errors="coerce").fillna(0.0)))
                    if var == 0.0:
                        print(f"  [warn] {m} produced identical scores for all candidates (variance=0).")
            except Exception:
                pass

if __name__ == "__main__":
    main()
