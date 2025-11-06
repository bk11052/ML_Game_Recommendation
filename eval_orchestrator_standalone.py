# -*- coding: utf-8 -*-
"""
eval_orchestrator_standalone.py
- 패키지 import 없이 돌아가는 '독립형' 오케스트레이터
- 아이디어: 모든 유틸/메트릭 함수를 내부에 포함하고,
  wrappers/ (또는 models/)를 subprocess로 호출하여 예측 CSV를 표준 스키마로 생성하게 함.

실행 예:
python eval_orchestrator_standalone.py \
  --preprocessed data/preprocessed_data.csv \
  --games_csv data/games.csv \
  --models hybrid cf_bias ibcf \
  --k_list 5 10 20 \
  --seed 42 \
  --test_ratio 0.2
"""

import argparse, sys, subprocess, os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# ------------------------------
# 경로 유틸
# ------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent  # 파일이 루트에 있을 것을 권장
WRAPPERS_DIR = PROJECT_ROOT / "wrappers"
MODELS_DIR   = PROJECT_ROOT / "models"
DEFAULT_OUT  = PROJECT_ROOT / "eval_outputs"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 데이터 로드
# ------------------------------
def load_preprocessed(preprocessed_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfp = pd.read_csv(preprocessed_csv)
    need = {"author_id", "app_id", "is_positive_encoded", "title"}
    missing = need - set(dfp.columns)
    if missing:
        raise ValueError(f"[DATA] Missing columns in {preprocessed_csv}: {missing}")

    ratings = (
        dfp[["author_id","app_id","is_positive_encoded"]]
        .rename(columns={"is_positive_encoded":"rating"})
        .dropna(subset=["author_id","app_id","rating"])
        .astype({"rating": int})
        .drop_duplicates(["author_id","app_id"], keep="last")
        .reset_index(drop=True)
    )
    games = (
        dfp[["app_id","title"]]
        .drop_duplicates("app_id", keep="last")
        .reset_index(drop=True)
    )
    return ratings, games

def load_raw(review_csv: Path, games_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dr = pd.read_csv(review_csv)
    dg = pd.read_csv(games_csv)

    if "is_positive" not in dr.columns:
        raise ValueError("[DATA] review.csv must contain 'is_positive'")

    ratings = (
        dr.assign(rating=lambda x: x["is_positive"].astype(str).str.lower().str.startswith("pos").astype(int))
          [["author_id","app_id","rating"]]
          .dropna(subset=["author_id","app_id","rating"])
          .astype({"rating": int})
          .drop_duplicates(["author_id","app_id"], keep="last")
          .reset_index(drop=True)
    )
    if "title" not in dg.columns:
        raise ValueError("[DATA] games.csv must contain 'title'")
    games = dg[["app_id","title"]].drop_duplicates("app_id").reset_index(drop=True)
    return ratings, games

# ------------------------------
# Split / GT
# ------------------------------
def user_holdout_split(ratings: pd.DataFrame, test_ratio: float=0.2, seed: int=42):
    rs = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for uid, grp in ratings.groupby("author_id"):
        idx = grp.index.to_numpy()
        if len(idx) == 1:
            tr_idx += idx.tolist(); continue
        n_test = max(1, int(round(len(idx) * test_ratio)))
        te = set(rs.choice(idx, size=n_test, replace=False).tolist())
        for i in idx:
            (te_idx if i in te else tr_idx).append(i)
    train = ratings.loc[tr_idx].reset_index(drop=True)
    test  = ratings.loc[te_idx].reset_index(drop=True)
    return train, test

def build_ground_truth(test_df: pd.DataFrame) -> Dict[str, set]:
    return {u: set(g["app_id"].tolist()) for u, g in test_df.groupby("author_id")}

# ------------------------------
# Metrics
# ------------------------------
def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2.0 * p * r / (p + r)

def compute_user_metrics_at_k(pred_items: List, gt_items: set, K: int):
    if K <= 0 or len(gt_items) == 0:
        return 0.0, 0.0, 0.0
    topk = pred_items[:K]
    hits = sum(1 for x in topk if x in gt_items)
    P = hits / float(K)
    R = hits / float(len(gt_items))
    return P, R, f1(P, R)

def macro_avg(series: pd.Series) -> Tuple[float, float]:
    return (0.0, 0.0) if series.empty else (float(series.mean()), float(series.std(ddof=0)))

def eval_one_model(pred_csv: Path, test_df: pd.DataFrame, k_list: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.read_csv(pred_csv)
    need = {"author_id","app_id","score","rank"}
    missing = need - set(preds.columns)
    if missing:
        raise SystemExit(f"[EVAL] predictions schema mismatch in {pred_csv.name}: missing {missing}")

    gt = build_ground_truth(test_df)
    rows = []
    preds_sorted = preds.sort_values(["author_id","score","rank"], ascending=[True, False, True])

    for uid, grp in preds_sorted.groupby("author_id"):
        if uid not in gt or len(gt[uid]) == 0:
            continue
        rec_list = grp["app_id"].tolist()
        row = {"author_id": uid}
        for K in k_list:
            p, r, ff = compute_user_metrics_at_k(rec_list, gt[uid], K)
            row[f"P@{K}"]  = p
            row[f"R@{K}"]  = r
            row[f"F1@{K}"] = ff
        rows.append(row)
    per_user = pd.DataFrame(rows)

    summ_rows = []
    for K in k_list:
        if per_user.empty:
            pm=ps=rm=rs=fm=fs=0.0
        else:
            pm, ps = macro_avg(per_user[f"P@{K}"])
            rm, rs = macro_avg(per_user[f"R@{K}"])
            fm, fs = macro_avg(per_user[f"F1@{K}"])
        summ_rows.append({"K":K,"P_mean":pm,"P_std":ps,"R_mean":rm,"R_std":rs,"F1_mean":fm,"F1_std":fs})
    summary = pd.DataFrame(summ_rows)
    return per_user, summary

# ------------------------------
# 모델 실행 (wrappers 우선, 실패 시 models 직접 호출)
# ------------------------------
def run_model(model_key: str, train_csv: Path, test_csv: Path, games_csv: Path, out_csv: Path, topk: int=200):
    """
    model_key ∈ {"hybrid","cf_bias","ibcf"}
    1) wrappers/wrap_*.py 존재 → 그걸 호출
    2) 없으면 models/*.py 직접 호출 (모델 파일에 --out 저장 로직이 필요)
    """
    ensure_dir(out_csv.parent)

    py = sys.executable
    wrapper_map = {
        "hybrid": WRAPPERS_DIR / "wrap_hybrid.py",
        "cf_bias": WRAPPERS_DIR / "wrap_cf_bias.py",
        "ibcf": WRAPPERS_DIR / "wrap_ibcf.py",
    }
    model_map = {
        "hybrid": MODELS_DIR / "Hybrid.py",
        "cf_bias": MODELS_DIR / "CFwithBias.py",
        "ibcf": MODELS_DIR / "IBCF.py",
    }

    # 1) wrapper 우선
    w = wrapper_map.get(model_key)
    if w and w.exists():
        cmd = [
            py, str(w),
            "--train_csv", str(train_csv),
            "--test_csv",  str(test_csv),
            "--games_csv", str(games_csv),
            "--topk",      str(topk),
            "--out",       str(out_csv),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode == 0:
            print(cp.stdout.strip() or f"[{model_key}] wrapper OK → {out_csv.name}")
            return
        else:
            print(cp.stdout)
            print(cp.stderr, file=sys.stderr)
            print(f"[{model_key}] wrapper failed, trying models/*.py ...", file=sys.stderr)

    # 2) models 직접 호출 (모델 파일이 CLI 인자 지원해야 함)
    m = model_map.get(model_key)
    if not m or not m.exists():
        raise SystemExit(f"[RUN] {model_key}: neither wrapper nor model file found.")

    cmd2 = [
        py, str(m),
        "--train_csv", str(train_csv),
        "--test_csv",  str(test_csv),
        "--games_csv", str(games_csv),
        "--topk",      str(topk),
        "--out",       str(out_csv),
    ]
    cp2 = subprocess.run(cmd2, capture_output=True, text=True)
    if cp2.returncode != 0:
        print(cp2.stdout)
        print(cp2.stderr, file=sys.stderr)
        raise SystemExit(f"[RUN] {model_key}: model script failed.")
    print(cp2.stdout.strip() or f"[{model_key}] model OK → {out_csv.name}")

# ------------------------------
# 메인
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed", default=str(PROJECT_ROOT / "data" / "preprocessed_data.csv"))
    ap.add_argument("--review_csv",  default=str(PROJECT_ROOT / "data" / "review.csv"))
    ap.add_argument("--games_csv",   default=str(PROJECT_ROOT / "data" / "games.csv"))
    ap.add_argument("--use_preprocessed_only", action="store_true")
    ap.add_argument("--outdir",      default=str(DEFAULT_OUT))
    ap.add_argument("--models", nargs="+", default=["hybrid","cf_bias","ibcf"],
                    choices=["hybrid","cf_bias","ibcf"])
    ap.add_argument("--k_list", nargs="+", type=int, default=[5,10,20])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=200)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    pre = Path(args.preprocessed)
    if pre.exists():
        print(f"[DATA] Using preprocessed: {pre}")
        ratings, games = load_preprocessed(pre)
    else:
        if args.use_preprocessed_only:
            raise SystemExit("[DATA] preprocessed_data.csv not found (fallback disabled).")
        print(f"[DATA] Fallback to raw: {args.review_csv}, {args.games_csv}")
        ratings, games = load_raw(Path(args.review_csv), Path(args.games_csv))

    # Split
    train_df, test_df = user_holdout_split(ratings, test_ratio=args.test_ratio, seed=args.seed)
    train_csv = outdir / "train_eval.csv"
    test_csv  = outdir / "test_eval.csv"
    games_csv = Path(args.games_csv)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"[SPLIT] train={len(train_df)}  test={len(test_df)}  users(train)={train_df.author_id.nunique()}")

    # Run models (wrappers → models fallback)
    pred_paths = {}
    for m in args.models:
        out_csv = outdir / f"{m}_predictions.csv"
        print(f"[RUN] {m} ...")
        run_model(m, train_csv, test_csv, games_csv, out_csv, topk=args.topk)
        pred_paths[m] = out_csv

    # Evaluate
    per_all, sum_all = [], []
    for m in args.models:
        per, summ = eval_one_model(pred_paths[m], test_df, args.k_list)
        if not per.empty:
            per["model"] = m
            per_all.append(per)
        summ["model"] = m
        sum_all.append(summ)

    if per_all:
        pd.concat(per_all, ignore_index=True).to_csv(outdir / "per_user_metrics.csv", index=False)
    pd.concat(sum_all, ignore_index=True).to_csv(outdir / "summary_metrics.csv", index=False)
    print(f"[DONE] metrics saved → {outdir}")

if __name__ == "__main__":
    # 어떤 작업 디렉터리에서 실행하든 항상 프로젝트 루트 기준으로 경로가 동작하도록 보장
    os.chdir(PROJECT_ROOT)
    main()
