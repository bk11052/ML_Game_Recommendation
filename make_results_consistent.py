#!/usr/bin/env python3
"""
make_results_consistent.py

Creates a canonical train/test split and evaluates all requested models
using the same candidates, same ranking/normalization rules, and writes
canonical prediction CSV(s) and evaluation outputs.

Produces:
 - canonical/train_split.csv
 - canonical/test_split.csv
 - canonical/test_positive.csv
 - canonical/predictions_all_models.csv
 - canonical/per_user_metrics.csv
 - canonical/summary_metrics.csv
 - canonical/plots/*.png

Usage:
  python make_results_consistent.py --data preprocessed_data.csv --out canonical --ks 5 10 20 --sample_users 300

This script avoids editing other scripts: instead it produces canonical outputs
that other scripts can consume for identical evaluation results.
"""
import os, argparse, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import your adapters if available
from adapters import CBF_TFIDF, MF_Simple, HybridWeighted

# Simple FM and WordEmb implementations (lightweight)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def per_user_temporal_holdout(df, time_col="id"):
    if time_col not in df.columns:
        df = df.reset_index().rename(columns={"index":"id"})
    df = df.sort_values(["author_id", time_col]).copy()
    train_idx, test_idx = [], []
    for uid, g in df.groupby("author_id", sort=False):
        if len(g) == 1:
            test_idx.extend(g.index.tolist())
        else:
            test_idx.append(g.index.tolist()[-1])
            train_idx.extend(g.index.tolist()[:-1])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)

def f1_at_k(y_true_set, ranked, k):
    topk = ranked[:k]
    hits = sum(1 for x in topk if x in y_true_set)
    prec = hits / max(len(topk),1)
    rec  = hits / max(len(y_true_set),1) if y_true_set else 0.0
    return (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0

def ndcg_at_k(y_true_set, ranked, k):
    dcg = 0.0
    for i,it in enumerate(ranked[:k], start=1):
        rel = 1.0 if it in y_true_set else 0.0
        dcg += rel / math.log2(i+1)
    ideal = min(k, len(y_true_set))
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i+1) for i in range(1, ideal+1))
    return dcg / idcg if idcg>0 else 0.0

# --- Lightweight FM ---
class SimpleFM:
    def __init__(self, n_factors=8, lr=0.05, epochs=6, seed=SEED):
        self.n_factors = n_factors
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
    def fit(self, train_df):
        users = train_df["author_id"].astype(str).unique().tolist()
        items = train_df["app_id"].astype(str).unique().tolist()
        self.u2i = {u:i for i,u in enumerate(users)}
        self.i2i = {a:i for i,a in enumerate(items)}
        nU, nI = len(users), len(items)
        rng = np.random.default_rng(self.seed)
        self.w0 = 0.0
        self.Wu = np.zeros(nU); self.Wi = np.zeros(nI)
        self.Vu = rng.normal(scale=0.1, size=(nU, self.n_factors))
        self.Vi = rng.normal(scale=0.1, size=(nI, self.n_factors))
        dfu = train_df.drop_duplicates(subset=["author_id","app_id"]).copy()
        rows = dfu["author_id"].map(self.u2i).values
        cols = dfu["app_id"].map(self.i2i).values
        vals = dfu["is_positive_encoded"].astype(float).values
        for ep in range(self.epochs):
            for u,i,y in zip(rows, cols, vals):
                pred = self.w0 + self.Wu[u] + self.Wi[i] + np.sum(self.Vu[u]*self.Vi[i])
                e = y - pred
                self.w0 += self.lr * e
                self.Wu[u] += self.lr * (e - 1e-4 * self.Wu[u])
                self.Wi[i] += self.lr * (e - 1e-4 * self.Wi[i])
                v_u = self.Vu[u]; v_i = self.Vi[i]
                self.Vu[u] += self.lr * (e * v_i - 1e-4 * v_u)
                self.Vi[i] += self.lr * (e * v_u - 1e-4 * v_i)
    def score_series(self, user_id, cands):
        if user_id not in self.u2i:
            return pd.Series(0.0, index=cands)
        u = self.u2i[user_id]
        scores = {}
        for a in cands:
            j = self.i2i.get(a, None)
            if j is None: scores[a]=0.0
            else: scores[a] = self.w0 + self.Wu[u] + self.Wi[j] + float(np.sum(self.Vu[u]*self.Vi[j]))
        s = pd.Series(scores)
        if s.max()!=s.min(): s=(s-s.min())/(s.max()-s.min())
        return s
    def recommend(self, user_id, cands, topk=20):
        return [(app,float(score)) for app,score in self.score_series(user_id,cands).sort_values(ascending=False).head(topk).items()]

# --- WordEmb (TFIDF + SVD) ---
class WordEmbApprox:
    def __init__(self, n_components=50):
        self.n_components = n_components
    def fit(self, item_titles_series):
        ids = item_titles_series.index.astype(str).tolist()
        docs = item_titles_series.fillna("").astype(str).tolist()
        vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X = vect.fit_transform(docs)
        n_comp = min(self.n_components, max(1, X.shape[1]-1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        self.emb = self.svd.fit_transform(X)
        self.appids = ids
        self.map = {a:i for i,a in enumerate(self.appids)}
    def score_series(self, user_id, cands, train_df):
        liked = train_df[(train_df["author_id"]==user_id) & (train_df["is_positive_encoded"]==1)]["app_id"].astype(str).unique().tolist()
        if not liked: return pd.Series(0.0, index=cands)
        liked_idx = [self.map[a] for a in liked if a in self.map]
        if not liked_idx: return pd.Series(0.0, index=cands)
        user_emb = np.mean(self.emb[liked_idx], axis=0)
        scores={}
        for a in cands:
            j = self.map.get(a,None)
            if j is None: scores[a]=0.0
            else:
                scores[a] = float(np.dot(user_emb, self.emb[j])/(np.linalg.norm(user_emb)*np.linalg.norm(self.emb[j])+1e-8))
        s = pd.Series(scores)
        if s.max()!=s.min(): s=(s-s.min())/(s.max()-s.min())
        return s
    def recommend(self, user_id, cands, topk=20, train_df=None):
        return [(app,float(score)) for app,score in self.score_series(user_id,cands,train_df).sort_values(ascending=False).head(topk).items()]

def run_all(data_path, outdir, ks=[5,10,20], sample_users=300, candidate_limit=2000):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(data_path, dtype={"author_id":str,"app_id":str})
    # canonical split (save to disk so other scripts can reuse)
    train, test = per_user_temporal_holdout(df, time_col="id")
    train.to_csv(os.path.join(outdir,"train_split.csv"), index=False)
    test.to_csv(os.path.join(outdir,"test_split.csv"), index=False)
    # positive test file for compatibility
    test[test["is_positive_encoded"]==1][["author_id","app_id"]].to_csv(os.path.join(outdir,"test_positive.csv"), index=False)

    # canonical candidate list: sorted to ensure deterministic order
    all_items = sorted(df["app_id"].unique().astype(str).tolist())

    # instantiate models (CBF/CF from adapters + our FM/WordEmb)
    item_titles = df.drop_duplicates(subset=["app_id"]).set_index("app_id")["title"]
    cbf = CBF_TFIDF(); cbf.fit(train, item_titles)
    cf  = MF_Simple(); cf.fit(train, item_titles)
    fm  = SimpleFM(n_factors=8, lr=0.05, epochs=6); fm.fit(train)
    we  = WordEmbApprox(n_components=30); we.fit(item_titles)

    # Hybrid simple alpha=0.3 (CF weight = 0.3)
    try:
        hyb_simple = HybridWeighted(cf, cbf, alpha=0.3)
        def hyb_simple_recomm(uid, cands, topk=20): return hyb_simple.recommend(uid, cands, topk=topk)
    except Exception:
        # wrapper that normalizes scores then weighted sum
        def hyb_simple_recomm(uid, cands, topk=20):
            s_cf = cf.score_series(uid, cands); s_cbf = cbf.score_series(uid, cands)
            if len(s_cf)>0: s_cf = (s_cf - s_cf.min())/(s_cf.max()-s_cf.min()+1e-9)
            if len(s_cbf)>0: s_cbf = (s_cbf - s_cbf.min())/(s_cbf.max()-s_cbf.min()+1e-9)
            s = 0.3 * s_cf + 0.7 * s_cbf
            return [(app,float(v)) for app,v in s.sort_values(ascending=False).head(topk).items()]

    # hybrid stacking: sample meta-train, logistic stacker
    sampled = train.sample(n=min(len(train),3000), random_state=SEED)
    meta = []
    for _,r in sampled.iterrows():
        uid, aid = r["author_id"], r["app_id"]
        s_cf = cf.score_series(uid, [aid]).get(aid,0.0) if hasattr(cf,"score_series") else 0.0
        s_cbf = cbf.score_series(uid, [aid]).get(aid,0.0) if hasattr(cbf,"score_series") else 0.0
        s_fm = fm.score_series(uid, [aid]).get(aid,0.0) if hasattr(fm,"score_series") else 0.0
        s_we = we.score_series(uid, [aid], train).get(aid,0.0)
        meta.append({"cf":s_cf,"cbf":s_cbf,"fm":s_fm,"we":s_we,"label":r["is_positive_encoded"]})
    meta_df = pd.DataFrame(meta).fillna(0.0)
    X = meta_df[["cf","cbf","fm","we"]].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    y = meta_df["label"].values
    stacker = LogisticRegression(max_iter=300, random_state=SEED).fit(Xs,y)

    def hybrid_stack_recomm(uid, cands, topk=20):
        s_cf = cf.score_series(uid, cands); s_cbf = cbf.score_series(uid, cands)
        s_fm = fm.score_series(uid, cands); s_we = we.score_series(uid, cands, train)
        df_s = pd.DataFrame({"cf":s_cf,"cbf":s_cbf,"fm":s_fm,"we":s_we}).fillna(0.0)
        if df_s.shape[0]==0:
            return []
        Xp = scaler.transform(df_s.values)
        probs = stacker.predict_proba(Xp)[:,1]
        s = pd.Series(probs, index=df_s.index)
        if s.max()!=s.min(): s=(s-s.min())/(s.max()-s.min())
        return [(app,float(v)) for app,v in s.sort_values(ascending=False).head(topk).items()]

    # models mapping: name -> function(uid, candidates, topk)
    models = {
        "CBF": lambda u,c,topk: cbf.recommend(u,c,topk),
        "CF":  lambda u,c,topk: cf.recommend(u,c,topk),
        "FM":  lambda u,c,topk: fm.recommend(u,c,topk),
        "WordEmb": lambda u,c,topk: we.recommend(u,c,topk, train_df=train),
        "Hybrid_simple_0.3_0.7": lambda u,c,topk: hyb_simple_recomm(u,c,topk),
        "Hybrid_stacking": lambda u,c,topk: hybrid_stack_recomm(u,c,topk)
    }

    # evaluation loop: deterministic user set (sorted)
    test_users = sorted(test["author_id"].unique().astype(str).tolist())
    if sample_users and sample_users>0:
        test_users = test_users[:sample_users]

    rows_pred=[]; per_rows=[]
    for model_name, fn in models.items():
        for uid in test_users:
            cands = [a for a in all_items if a not in set(train[train["author_id"]==uid]["app_id"].tolist())]
            # ensure deterministic ordering of candidates
            # apply candidate_limit if provided
            if candidate_limit and len(cands)>candidate_limit:
                cands = cands[:candidate_limit]
            recs = fn(uid, cands, topk=max(ks))
            ranked = [t[0] for t in recs]
            # store predictions with rank
            for r,app in enumerate(ranked, start=1):
                rows_pred.append({"model":model_name,"author_id":uid,"app_id":app,"rank":r,"score":recs[r-1][1] if r-1 < len(recs) else 0.0})
            ytrue = set(test[(test["author_id"]==uid) & (test["is_positive_encoded"]==1)]["app_id"].astype(str).tolist())
            for k in ks:
                per_rows.append({"model":model_name,"user_id":uid,"k":k,"f1":f1_at_k(ytrue,ranked,k),"ndcg":ndcg_at_k(ytrue,ranked,k)})

    pred_df = pd.DataFrame(rows_pred).sort_values(["model","author_id","rank"])
    pred_df.to_csv(os.path.join(outdir,"predictions_all_models.csv"), index=False)
    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(os.path.join(outdir,"per_user_metrics.csv"), index=False)
    summary = per_df.groupby(["model","k"])[["f1","ndcg"]].mean().reset_index()
    summary.to_csv(os.path.join(outdir,"summary_metrics.csv"), index=False)

    # plots
    ks_sorted = sorted(ks)
    plt.figure(figsize=(8,4))
    for m in summary["model"].unique():
        tmp = summary[summary["model"]==m].set_index("k").reindex(ks_sorted)
        plt.plot(ks_sorted, tmp["f1"], marker="o", label=m)
    plt.title("F1@K"); plt.xlabel("k"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir,"F1_K.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,4))
    for m in summary["model"].unique():
        tmp = summary[summary["model"]==m].set_index("k").reindex(ks_sorted)
        plt.plot(ks_sorted, tmp["ndcg"], marker="o", label=m)
    plt.title("nDCG@K"); plt.xlabel("k"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir,"nDCG_K.png"), bbox_inches="tight")
    plt.close()

    print("[done] canonical outputs saved to:", outdir)
    return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed_data.csv")
    ap.add_argument("--out", default="canonical")
    ap.add_argument("--ks", nargs="+", type=int, default=[5,10,20])
    ap.add_argument("--sample_users", type=int, default=300)
    ap.add_argument("--candidate_limit", type=int, default=2000)
    args = ap.parse_args()
    run_all(args.data, args.out, ks=args.ks, sample_users=args.sample_users, candidate_limit=args.candidate_limit)
