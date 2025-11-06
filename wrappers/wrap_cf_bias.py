# -*- coding: utf-8 -*-
# ============================================================
# wrap_cf_bias.py
#  - Thin wrapper for models/CFwithBias.py
#  - Input : --train_csv --test_csv --games_csv --topk --out
#  - Output: CSV schema [author_id, app_id, score, rank]
# ------------------------------------------------------------
# Author: Team ML_Game_Recommendation (Evaluation)
# ============================================================

import argparse
from pathlib import Path
import pandas as pd
import subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--games_csv", required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--topk",      type=int, default=200)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 1) 함수형 API 우선
    try:
        from models.CFwithBias import run_cf_with_bias
        train = pd.read_csv(args.train_csv)
        test  = pd.read_csv(args.test_csv)
        games = pd.read_csv(args.games_csv)
        users = sorted(test["author_id"].unique().tolist())
        preds = run_cf_with_bias(train, games, users, topk=args.topk)
        need = {"author_id","app_id","score","rank"}
        if not need.issubset(preds.columns):
            raise RuntimeError(f"[wrap_cf_bias] run_cf_with_bias 반환 스키마 불일치: {need - set(preds.columns)}")
        preds.to_csv(args.out, index=False, encoding="utf-8")
        print(f"[wrap_cf_bias] saved -> {args.out}")
        return
    except Exception as e:
        print(f"[wrap_cf_bias] fallback to CLI: {e}", file=sys.stderr)

    # 2) CLI 호출
    cmd = [
        sys.executable, "models/CFwithBias.py",
        "--train_csv", args.train_csv,
        "--test_csv",  args.test_csv,
        "--games_csv", args.games_csv,
        "--topk",      str(args.topk),
        "--out",       args.out,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr, file=sys.stderr)
        raise SystemExit("[wrap_cf_bias] CFwithBias.py 실행 실패 (CLI).")
    print(cp.stdout.strip() or f"[wrap_cf_bias] done -> {args.out}")

if __name__ == "__main__":
    main()
