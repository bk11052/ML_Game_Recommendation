# evaluate_requested_models.py
"""
Evaluate requested models:
 - CBF (adapters.CBF_TFIDF)
 - CF  (adapters.MF_Simple)
 - FM  (SimpleFM implemented here)
 - WordEmb (TF-IDF + TruncatedSVD based approximation)
 - Hybrid_simple_0.3_0.7 (CF*0.3 + CBF*0.7)
 - Hybrid_simple_0.5_0.5 (CF*0.5 + CBF*0.5)
 - Hybrid_stacking (meta-learner logistic on CF/CBF/FM/WordEmb scores)

Usage:
  python evaluate_requested_models.py --data preprocessed_data.csv --outdir eval_all --ks 5 10 20 --sample_users 300 --candidate_limit 2000
"""
import os
import math
import argparse
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# try to import user's adapters (CBF_TFIDF, MF_Simple)
try:
    from adapters import CBF_TFIDF, MF_Simple, HybridWeighted
except Exception as e:
    raise ImportError("Could not import adapters.py with CBF_TFIDF and MF_Simple. Make sure adapters.py is in the same folder.") from e

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# metrics
# -------------------------
def f1_at_k(y_true_set, ranked_items, k):
    topk = ranked_items[:k]
    hits = sum(1 for x in topk if x in y_true_set)
    prec = hits / max(len(topk), 1)
    rec  = hits / max(len(y_true_set), 1) if y_true_set else 0.0
    return (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0

def ndcg_at_k(y_true_set, ranked_items, k):
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k], start=1):
        rel = 1.0 if item in y_true_set else 0.0
        dcg += rel / math.log2(i+1)
    ideal = min(k, len(y_true_set))
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i+1) for i in range(1, ideal+1))
    return dcg / idcg if idcg>0 else 0.0

# -------------------------
# simple FM (lightweight)
# -------------------------
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
    def score_series(self, user_id, candidate_items):
        if user_id not in self.u2i:
            return pd.Series(0.0, index=candidate_items)
        u = self.u2i[user_id]
        scores = {}
        for app in candidate_items:
            i = self.i2i.get(app, None)
            if i is None:
                scores[app] = 0.0
            else:
                scores[app] = self.w0 + self.Wu[u] + self.Wi[i] + float(np.sum(self.Vu[u]*self.Vi[i]))
        s = pd.Series(scores)
        if s.max() != s.min():
            s = (s - s.min())/(s.max()-s.min())
        return s
    def recommend(self, user_id, candidate_items, topk=20):
        return [(app,float(score)) for app,score in self.score_series(user_id,candidate_items).sort_values(ascending=False).head(topk).items()]

# -------------------------
# Word-embedding approx (TFIDF + SVD)
# -------------------------
class WordEmbApprox:
    def __init__(self, n_components=50):
        self.n_components = n_components
    def fit(self, item_titles_series):
        # item_titles_series: Series indexed by app_id with title text
        ids = item_titles_series.index.astype(str).tolist()
        docs = item_titles_series.fillna("").astype(str).tolist()
        vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X = vect.fit_transform(docs)
        n_comp = min(self.n_components, max(1, X.shape[1]-1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        self.emb = self.svd.fit_transform(X)
        self.appids = ids
        self.map = {a:i for i,a in enumerate(self.appids)}
    def score_series(self, user_id, candidate_items, train_df):
        # collect liked items (positive) for the user from train_df
        liked = train_df[(train_df["author_id"]==user_id) & (train_df["is_positive_encoded"]==1)]["app_id"].astype(str).unique().tolist()
        if not liked:
            return pd.Series(0.0, index=candidate_items)
        liked_idx = [self.map[a] for a in liked if a in self.map]
        if not liked_idx:
            return pd.Series(0.0, index=candidate_items)
        user_emb = np.mean(self.emb[liked_idx], axis=0)
        scores={}
        for app in candidate_items:
            j = self.map.get(app,None)
            if j is None:
                scores[app]=0.0
            else:
                # cosine similarity
                denom = (np.linalg.norm(user_emb)*np.linalg.norm(self.emb[j])+1e-8)
                scores[app] = float(np.dot(user_emb, self.emb[j]) / denom)
        s = pd.Series(scores)
        if s.max() != s.min():
            s = (s - s.min())/(s.max()-s.min())
        return s
    def recommend(self, user_id, candidate_items, topk=20, train_df=None):
        return [(app,float(score)) for app,score in self.score_series(user_id,candidate_items, train_df).sort_values(ascending=False).head(topk).items()]

# -------------------------
# utility / evaluation runner
# -------------------------
def per_user_holdout(df, time_col="id"):
    if time_col not in df.columns:
        df = df.reset_index().rename(columns={"index":"id"})
    df = df.sort_values(["author_id", time_col])
    train_idx, test_idx = [], []
    for uid, g in df.groupby("author_id", sort=False):
        if len(g) == 1:
            test_idx.extend(g.index.tolist())
        else:
            test_idx.append(g.index.tolist()[-1])
            train_idx.extend(g.index.tolist()[:-1])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)

def evaluate_models(train, test, models_dict, ks=[5,10,20], candidate_limit=None, sample_users=None, outdir="eval_out"):
    os.makedirs(outdir, exist_ok=True)
    all_items = sorted(train["app_id"].unique().astype(str).tolist())
    interacted = train.groupby("author_id")["app_id"].apply(set).to_dict()
    users = sorted(test["author_id"].unique())
    if sample_users is not None and sample_users > 0:
        users = users[:sample_users]

    rows_pred = []
    per_rows = []
    for model_name, model in models_dict.items():
        print("Evaluating model:", model_name)
        for uid in users:
            cand = [i for i in all_items if i not in interacted.get(uid, set())]
            if candidate_limit and len(cand) > candidate_limit:
                cand = cand[:candidate_limit]

            # --- SAFELY call recommend() and pass train_df if provided in model dict ---
            model_train_df = None
            if isinstance(model, dict) and "train_df" in model:
                model_train_df = model["train_df"]
            obj = model["obj"] if isinstance(model, dict) and "obj" in model else model

            # try calling recommend with train_df kw if available, otherwise fallback
            try:
                if model_train_df is not None:
                    # some recommend implementations accept train_df kw; pass it
                    recs = obj.recommend(uid, cand, topk=max(ks), train_df=model_train_df)
                else:
                    recs = obj.recommend(uid, cand, topk=max(ks))
            except TypeError:
                # recommend() didn't accept train_df kw (or signature mismatch) -> call without it
                recs = obj.recommend(uid, cand, topk=max(ks))
            except Exception:
                # as a last resort, try score_series
                try:
                    s = obj.score_series(uid, cand)
                    ranked = list(s.sort_values(ascending=False).index)[:max(ks)]
                    recs = [(app, float(s.get(app,0.0))) for app in ranked]
                except Exception:
                    recs = []

            ranked = [t[0] for t in recs]

            # save predictions
            for r, app in enumerate(ranked, start=1):
                rows_pred.append({"model":model_name, "author_id":uid, "app_id":app, "rank":r, "score":recs[r-1][1] if r-1 < len(recs) else 0.0})

            # compute per-user metrics
            y_true = set(test[(test["author_id"]==uid) & (test["is_positive_encoded"]==1)]["app_id"].astype(str).tolist())
            for k in ks:
                per_rows.append({"model":model_name, "user_id":uid, "k":k, "f1":f1_at_k(y_true, ranked, k), "ndcg":ndcg_at_k(y_true, ranked, k)})

    pred_df = pd.DataFrame(rows_pred)
    per_df = pd.DataFrame(per_rows)
    summary = per_df.groupby(["model","k"])[["f1","ndcg"]].mean().reset_index()
    # save
    pred_df.to_csv(os.path.join(outdir, "predictions_all_models.csv"), index=False)
    per_df.to_csv(os.path.join(outdir, "per_user_metrics.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "summary_metrics.csv"), index=False)

    # plots
    ks_sorted = sorted(set(summary["k"].tolist()))
    plt.figure(figsize=(8,4))
    for m in summary["model"].unique():
        tmp = summary[summary["model"]==m].set_index("k").reindex(ks_sorted)
        plt.plot(ks_sorted, tmp["f1"], marker='o', label=m)
    plt.title("F1@K"); plt.xlabel("k"); plt.ylabel("F1"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir, "F1_K.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,4))
    for m in summary["model"].unique():
        tmp = summary[summary["model"]==m].set_index("k").reindex(ks_sorted)
        plt.plot(ks_sorted, tmp["ndcg"], marker='o', label=m)
    plt.title("nDCG@K"); plt.xlabel("k"); plt.ylabel("nDCG"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir, "nDCG_K.png"), bbox_inches="tight")
    plt.close()

    print("Saved outputs to", outdir)
    return pred_df, per_df, summary

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed_data.csv")
    ap.add_argument("--outdir", default="eval_all")
    ap.add_argument("--ks", nargs="+", type=int, default=[5,10,20])
    ap.add_argument("--sample_users", type=int, default=300, help="set to -1 or None for all users")
    ap.add_argument("--candidate_limit", type=int, default=2000, help="limit candidates to speed up")
    args = ap.parse_args()

    df = pd.read_csv(args.data, dtype={"author_id":str,"app_id":str})
    train, test = per_user_holdout(df, time_col="id")
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # prepare item titles (for WordEmb)
    item_titles = df.drop_duplicates(subset=["app_id"]).set_index("app_id")["title"] if "title" in df.columns else pd.Series(dtype=str)

    # instantiate models
    cbf = CBF_TFIDF(); cbf.fit(train, item_titles)
    cf  = MF_Simple(); cf.fit(train, item_titles)

    fm = SimpleFM(n_factors=8, lr=0.05, epochs=6); fm.fit(train)

    we = WordEmbApprox(n_components=30)
    # ensure item_titles is non-empty when possible
    if item_titles.shape[0] > 0:
        we.fit(item_titles)
    else:
        # if no titles, we still keep we but it will return zeros
        print("Warning: item_titles is empty; WordEmb will produce zero scores.")

    # Hybrid simple alpha=0.3
    try:
        hyb_simple = HybridWeighted(cf, cbf, alpha=0.3)
        hyb_simple_obj = {"obj": hyb_simple}
    except Exception:
        class HybSimpleWrapper:
            def __init__(self, cf, cbf, alpha=0.3):
                self.cf = cf; self.cbf = cbf; self.alpha = alpha
            def score_series(self, uid, cands):
                s_cf = self.cf.score_series(uid, cands)
                s_cbf = self.cbf.score_series(uid, cands)
                a = s_cf; b = s_cbf
                if len(a) > 0 and a.max() != a.min():
                    a = (a - a.min())/(a.max()-a.min()+1e-9)
                if len(b) > 0 and b.max() != b.min():
                    b = (b - b.min())/(b.max()-b.min()+1e-9)
                s = self.alpha*a + (1.0-self.alpha)*b
                return s
            def recommend(self, uid, cands, topk=20):
                s = self.score_series(uid, cands)
                return [(app,float(score)) for app,score in s.sort_values(ascending=False).head(topk).items()]
        hyb_simple_obj = {"obj": HybSimpleWrapper(cf, cbf, alpha=0.3)}

    # Hybrid simple alpha = 0.5 (new)
    class HybWrapperAlpha:
        def __init__(self, cf, cbf, alpha=0.5):
            self.cf = cf
            self.cbf = cbf
            self.alpha = alpha
        def score_series(self, uid, cands):
            a = self.cf.score_series(uid, cands)
            b = self.cbf.score_series(uid, cands)
            a = a.reindex(cands).fillna(0.0)
            b = b.reindex(cands).fillna(0.0)
            if len(a) > 0 and a.max() != a.min():
                a = (a - a.min())/(a.max()-a.min()+1e-9)
            if len(b) > 0 and b.max() != b.min():
                b = (b - b.min())/(b.max()-b.min()+1e-9)
            s = self.alpha * a + (1.0 - self.alpha) * b
            return s
        def recommend(self, uid, cands, topk=20):
            s = self.score_series(uid, cands)
            return [(app,float(score)) for app,score in s.sort_values(ascending=False).head(topk).items()]

    hyb_05_obj = {"obj": HybWrapperAlpha(cf, cbf, alpha=0.5)}

    # Hybrid stacking: build meta-features on sampled interactions
    sampled = train.sample(n=min(len(train), 3000), random_state=SEED)
    meta_rows = []
    for _, r in sampled.iterrows():
        uid = r["author_id"]; aid = r["app_id"]
        try: s_cf = cf.score_series(uid, [aid]).get(aid,0.0)
        except: s_cf = 0.0
        try: s_cbf = cbf.score_series(uid, [aid]).get(aid,0.0)
        except: s_cbf = 0.0
        try: s_fm = fm.score_series(uid, [aid]).get(aid,0.0)
        except: s_fm = 0.0
        try: s_we = we.score_series(uid, [aid], train).get(aid,0.0)
        except: s_we = 0.0
        meta_rows.append({"cf":s_cf,"cbf":s_cbf,"fm":s_fm,"we":s_we,"label":r.get("is_positive_encoded",0)})
    meta_df = pd.DataFrame(meta_rows).fillna(0.0)
    if meta_df.shape[0] == 0:
        # fallback: if no meta rows, create dummy stacker that averages scores
        class DummyStacker:
            def predict_proba(self, X):
                probs = np.mean(X, axis=1)
                return np.vstack([1-probs, probs]).T
        scaler = StandardScaler()
        scaler.fit(np.zeros((1,4)))
        stacker = DummyStacker()
    else:
        X = meta_df[["cf","cbf","fm","we"]].values
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        y = meta_df["label"].values
        stacker = LogisticRegression(max_iter=300, random_state=SEED)
        stacker.fit(Xs,y)

    class HybridStackingWrapper:
        def __init__(self, cf, cbf, fm, we, stacker, scaler):
            self.cf=cf; self.cbf=cbf; self.fm=fm; self.we=we; self.stacker=stacker; self.scaler=scaler
        def score_series(self, uid, cands):
            s_cf = self.cf.score_series(uid, cands)
            s_cbf = self.cbf.score_series(uid, cands)
            s_fm = self.fm.score_series(uid, cands)
            s_we = self.we.score_series(uid, cands, train)
            df = pd.DataFrame({"cf":s_cf, "cbf":s_cbf, "fm":s_fm, "we":s_we}).fillna(0.0)
            if df.shape[0] == 0:
                return pd.Series(0.0, index=cands)
            Xp = self.scaler.transform(df.values)
            probs = self.stacker.predict_proba(Xp)[:,1]
            s = pd.Series(probs, index=df.index)
            if s.max()!=s.min():
                s=(s-s.min())/(s.max()-s.min())
            return s
        def recommend(self, uid, cands, topk=20):
            return [(app,float(score)) for app,score in self.score_series(uid,cands).sort_values(ascending=False).head(topk).items()]

    hybrid_stack = HybridStackingWrapper(cf, cbf, fm, we, stacker, scaler)

    # assemble models dict (pass train to WordEmb)
    models = {
        "CBF": {"obj":cbf},
        "CF":  {"obj":cf},
        "FM":  {"obj":fm},
        "WordEmb": {"obj": we, "train_df": train},      # <-- train_df provided here
        "Hybrid_simple_0.3_0.7": hyb_simple_obj,
        "Hybrid_simple_0.5_0.5": hyb_05_obj,
        "Hybrid_stacking": {"obj": hybrid_stack}
    }

    sample_users_arg = None if (args.sample_users is None or args.sample_users < 0) else args.sample_users
    pred, per_user, summary = evaluate_models(train, test, models, ks=args.ks, candidate_limit=args.candidate_limit, sample_users=sample_users_arg, outdir=outdir)
    print("SUMMARY:\n", summary)
    print("Done.")

if __name__ == "__main__":
    main()
