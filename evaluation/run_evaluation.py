import os, pandas as pd
from collections import defaultdict
from split import make_splits
from adapters import build_model_dict
from metrics import compute_topk

# =========================================================
# 1. 평가 실행 스크립트
# =========================================================
# - train/valid/test 스플릿 생성
# - 모델(popularity, hybrid, IBCF)별 Precision/Recall/nDCG/MAP 계산
# =========================================================

OUT = "evaluation/results"; os.makedirs(OUT, exist_ok=True)
SPL = "evaluation/splits"

def _ensure_splits():
    needed = ["train_user_holdout.csv","valid_user_holdout.csv","test_user_holdout.csv"]
    if not all(os.path.exists(os.path.join(SPL,n)) for n in needed):
        make_splits(preprocessed_path="preprocessed_data.csv", outdir=SPL)

def _load_split(name): return pd.read_csv(os.path.join(SPL, name))

def _gt_from(df):
    gt = defaultdict(set)
    for _,r in df.iterrows():
        gt[str(r["user_id"])].add(str(r["app_id"]))
    return gt

def _eval(models, gt_users, Ks=(5,10,20)):
    all_rows=[]
    for name, fn in models.items():
        preds={}
        for u in gt_users:
            try:
                preds[u] = [str(x) for x in fn(u, max(Ks))]
            except Exception:
                preds[u] = []
        df = compute_topk(preds, gt_users, ks=Ks)
        df["model"]=name
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)

def main():
    print("\n===============================================")
    print("🔥 Evaluation 시작")
    print("===============================================")

    _ensure_splits()
    train = _load_split("train_user_holdout.csv")
    valid = _load_split("valid_user_holdout.csv")
    test  = _load_split("test_user_holdout.csv")

    # 모델 자동 감지
    models = build_model_dict(train)

    # =========================================================
    # 2. 유저 홀드아웃 기반 평가
    # =========================================================
    gt_users = _gt_from(test)
    res_holdout = _eval(models, gt_users, Ks=(5,10,20))
    res_holdout.to_csv(os.path.join(OUT,"metrics_user_holdout.csv"), index=False)
    print("✅ [user_holdout] 결과 저장 완료")
    print(res_holdout)

    # =========================================================
    # 3. 시간 기반 평가 (있을 경우)
    # =========================================================
    if os.path.exists(os.path.join(SPL,"train_time.csv")) and os.path.exists(os.path.join(SPL,"test_time.csv")):
        tr_t = _load_split("train_time.csv")
        te_t = _load_split("test_time.csv")
        if len(te_t):
            models_t = build_model_dict(tr_t)
            gt_t = _gt_from(te_t)
            res_time = _eval(models_t, gt_t, Ks=(5,10,20))
            res_time.to_csv(os.path.join(OUT,"metrics_time_split.csv"), index=False)
            print("✅ [time_split] 결과 저장 완료")
            print(res_time)
        else:
            print("⚠️ test_time 데이터가 비어있습니다. 시간 평가 생략.")
    else:
        print("ℹ️ time-based split 파일 없음 → 생략")

if __name__ == "__main__":
    main()
