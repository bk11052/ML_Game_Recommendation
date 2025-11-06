# build_predictions.py
import argparse, os
import pandas as pd
from adapters import CBF_TFIDF, MF_Simple, HybridWeighted

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["author_id", "app_id", "title", "is_positive_encoded"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df["author_id"] = df["author_id"].astype(str)
    df["app_id"] = df["app_id"].astype(str)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    return df

def per_user_temporal_holdout(df: pd.DataFrame):
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
    ap.add_argument("--data", type=str, default="preprocessed_data.csv")
    ap.add_argument("--out", type=str, default="predictions.csv")
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

    rows = []
    for uid in sorted(test["author_id"].unique()):
        candidates = list(all_items - interacted_train.get(uid, set()))
        for model_name, model in [("CBF", cbf), ("CF", cf), (f"Hybrid(a={args.alpha})", hyb)]:
            recs = model.recommend(uid, candidates, topk=args.k)
            for rank, (app, score) in enumerate(recs, start=1):
                rows.append({"model": model_name, "author_id": uid, "app_id": app, "rank": rank, "score": score})

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    pos = test[test["is_positive_encoded"] == 1][["author_id", "app_id"]].copy()
    pos.to_csv("test_positive.csv", index=False)
    print(f"Saved predictions to: {args.out}")
    print("Saved test positives to: test_positive.csv")

if __name__ == "__main__":
    main()
