import math, argparse, os
import pandas as pd
import matplotlib.pyplot as plt

# ----- metrics -----
def ndcg_at_k(y_true_set, ranked_items, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k], start=1):
        rel = 1.0 if item in y_true_set else 0.0
        dcg += rel / math.log2(i + 1)
    ideal = min(k, len(y_true_set))
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0

def f1_at_k(y_true_set, ranked_items, k: int) -> float:
    topk = ranked_items[:k]
    hits = sum(1 for x in topk if x in y_true_set)
    prec = hits / max(len(topk), 1)
    rec  = hits / max(len(y_true_set), 1) if y_true_set else 0.0
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

# ----- loaders -----
def load_ground_truth(test_csv: str):
    df = pd.read_csv(test_csv)
    if "is_positive_encoded" in df.columns:
        df = df[df["is_positive_encoded"] == 1][["author_id", "app_id"]]
    df["author_id"] = df["author_id"].astype(str)
    df["app_id"] = df["app_id"].astype(str)
    return df.groupby("author_id")["app_id"].apply(set).to_dict()

def load_predictions(pred_csv: str):
    df = pd.read_csv(pred_csv)
    df["author_id"] = df["author_id"].astype(str)
    df["app_id"] = df["app_id"].astype(str)
    if "rank" in df.columns:
        df = df.sort_values(["model", "author_id", "rank"])
    elif "score" in df.columns:
        df = df.sort_values(["model", "author_id", "score"], ascending=[True, True, False])
    preds = {}
    for (m, u), g in df.groupby(["model", "author_id"]):
        preds.setdefault(m, {})[u] = g["app_id"].tolist()
    return preds

# ----- evaluation -----
def evaluate(preds, gt, ks):
    rows = []
    users = set().union(*[set(u.keys()) for u in preds.values()]) & set(gt.keys())
    for model, u2r in preds.items():
        for uid in users:
            if uid not in u2r: continue
            ranked = u2r[uid]; y_true = gt[uid]
            for k in ks:
                rows.append({
                    "model": model, "user_id": uid, "k": k,
                    "f1": f1_at_k(y_true, ranked, k),
                    "ndcg": ndcg_at_k(y_true, ranked, k),
                })
    per_user = pd.DataFrame(rows)
    summary = (
        per_user.groupby(["model", "k"])[["f1", "ndcg"]].mean().reset_index()
        if not per_user.empty else pd.DataFrame(columns=["model", "k", "f1", "ndcg"])
    )
    return per_user, summary

# ----- plotting -----
def plot_results(summary, outdir):
    os.makedirs(outdir, exist_ok=True)
    for metric in ["f1", "ndcg"]:
        plt.figure(figsize=(7,5))
        for model in summary["model"].unique():
            sub = summary[summary["model"] == model]
            plt.plot(sub["k"], sub[metric], marker='o', label=model)
        plt.title(f"{metric.upper()}@K by Model")
        plt.xlabel("K"); plt.ylabel(metric.upper())
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{metric}_lines.png"), dpi=150)
        plt.close()

        # bar plot (K=10 기준)
        k_target = 10 if 10 in summary["k"].unique() else summary["k"].min()
        sub = summary[summary["k"] == k_target].sort_values(metric, ascending=False)
        plt.figure(figsize=(6,4))
        plt.bar(sub["model"], sub[metric])
        plt.title(f"{metric.upper()}@{k_target}")
        plt.xlabel("Model"); plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{metric}_bar_{k_target}.png"), dpi=150)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="test_positive.csv")
    ap.add_argument("--pred", default="predictions.csv")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--outdir", default="metrics")
    ap.add_argument("--figdir", default="figs")
    args = ap.parse_args()

    gt = load_ground_truth(args.gt)
    preds = load_predictions(args.pred)
    per_user, summary = evaluate(preds, gt, args.ks)

    os.makedirs(args.outdir, exist_ok=True)
    per_user.to_csv(os.path.join(args.outdir, "per_user_metrics.csv"), index=False)
    summary.to_csv(os.path.join(args.outdir, "summary_metrics.csv"), index=False)
    print("\n=== SUMMARY ===")
    print(summary)

    plot_results(summary, args.figdir)
    print(f"\n✅ Saved CSVs to '{args.outdir}', and plots to '{args.figdir}'.")

if __name__ == "__main__":
    main()
