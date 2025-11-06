# -*- coding: utf-8 -*-
# ============================================================
# wrap_hybrid.py
#  - Thin wrapper for models/Hybrid.py
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

    # 1) 함수형 API 우선 시도
    try:
        from models.Hybrid import run_hybrid  # 팀원 코드에 함수가 있다면 가장 간단
        train = pd.read_csv(args.train_csv)
        test  = pd.read_csv(args.test_csv)
        games = pd.read_csv(args.games_csv)
        users = sorted(test["author_id"].unique().tolist())
        preds = run_hybrid(train, games, users, topk=args.topk)  # -> DF[author_id, app_id, score, rank]
        need = {"author_id","app_id","score","rank"}
        if not need.issubset(preds.columns):
            raise RuntimeError(f"[wrap_hybrid] run_hybrid 반환 스키마 불일치: {need - set(preds.columns)}")
        preds.to_csv(args.out, index=False, encoding="utf-8")
        print(f"[wrap_hybrid] saved -> {args.out}")
        return
    except Exception as e:
        print(f"[wrap_hybrid] fallback to CLI: {e}", file=sys.stderr)

    # 2) 없거나 실패하면 모델 스크립트를 CLI로 호출 (모델쪽이 --out 저장을 지원해야 함)
    cmd = [
        sys.executable, "models/Hybrid.py",
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
        raise SystemExit("[wrap_hybrid] Hybrid.py 실행 실패 (CLI).")
    print(cp.stdout.strip() or f"[wrap_hybrid] done -> {args.out}")

if __name__ == "__main__":
    main()
