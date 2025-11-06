# adapters.py
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CBF_TFIDF:
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features

    def fit(self, train_df: pd.DataFrame, item_titles: pd.Series):
        item_df = item_titles.drop_duplicates().to_frame(name="title")
        item_df["app_id"] = item_df.index.astype(str)
        self.item_index_ = item_df["app_id"].tolist()
        self.appid_to_pos_ = {app: i for i, app in enumerate(self.item_index_)}
        self.vectorizer_ = TfidfVectorizer(
            stop_words="english", token_pattern=r"\b\w{2,}\b", max_features=self.max_features
        )
        self.tfidf_ = self.vectorizer_.fit_transform(item_df["title"].fillna(""))
        self.sim_ = cosine_similarity(self.tfidf_)
        self.user_pos_ = (
            train_df[train_df["is_positive_encoded"] == 1]
            .groupby("author_id")["app_id"]
            .apply(lambda s: list(map(str, s)))
            .to_dict()
        )

    def score_series(self, user_id: str, candidate_items: List[str]) -> pd.Series:
        liked = self.user_pos_.get(user_id, [])
        liked_idx = [self.appid_to_pos_.get(a) for a in liked if a in self.appid_to_pos_]
        liked_idx = [i for i in liked_idx if i is not None]
        if not liked_idx:
            return pd.Series(0.0, index=candidate_items)
        scores = {}
        for app in candidate_items:
            j = self.appid_to_pos_.get(app, None)
            scores[app] = float(np.sum(self.sim_[j, liked_idx])) if j is not None else 0.0
        s = pd.Series(scores)
        if s.max() > 0:
            s = s / s.max()
        return s

    def recommend(self, user_id: str, candidate_items: List[str], topk: int = 10):
        s = self.score_series(user_id, candidate_items).sort_values(ascending=False)
        return [(app, float(score)) for app, score in s.head(topk).items()]

class MF_Simple:
    def __init__(self, factors=20, lr=0.01, reg=0.01, epochs=15, seed=42):
        self.factors = factors; self.lr = lr; self.reg = reg; self.epochs = epochs; self.seed = seed

    def fit(self, train_df: pd.DataFrame, item_titles: pd.Series):
        users = train_df["author_id"].astype(str).unique().tolist()
        items = train_df["app_id"].astype(str).unique().tolist()
        self.user_to_idx_ = {u: i for i, u in enumerate(users)}
        self.item_to_idx_ = {a: i for i, a in enumerate(items)}
        self.idx_to_item_ = {i: a for a, i in self.item_to_idx_.items()}
        nU, nI = len(users), len(items)
        rng = np.random.default_rng(self.seed)
        self.P_ = rng.normal(scale=1.0 / self.factors, size=(nU, self.factors))
        self.Q_ = rng.normal(scale=1.0 / self.factors, size=(nI, self.factors))
        df = train_df.drop_duplicates(subset=["author_id", "app_id"]).copy()
        rows = df["author_id"].map(self.user_to_idx_).values
        cols = df["app_id"].map(self.item_to_idx_).values
        vals = df["is_positive_encoded"].astype(float).values
        for _ in range(self.epochs):
            for u, i, r in zip(rows, cols, vals):
                r_hat = float(np.dot(self.P_[u], self.Q_[i]))
                e = r - r_hat
                pu = self.P_[u].copy(); qi = self.Q_[i].copy()
                self.P_[u] += self.lr * (e * qi - self.reg * pu)
                self.Q_[i] += self.lr * (e * pu - self.reg * qi)

    def score_series(self, user_id: str, candidate_items: List[str]) -> pd.Series:
        if user_id not in self.user_to_idx_:
            return pd.Series(0.0, index=candidate_items)
        u = self.user_to_idx_[user_id]
        scores = {}
        for app in candidate_items:
            i = self.item_to_idx_.get(app)
            scores[app] = float(np.dot(self.P_[u], self.Q_[i])) if i is not None else 0.0
        s = pd.Series(scores)
        if s.max() > s.min():
            s = (s - s.min()) / (s.max() - s.min())
        return s

    def recommend(self, user_id: str, candidate_items: List[str], topk: int = 10):
        s = self.score_series(user_id, candidate_items).sort_values(ascending=False)
        return [(app, float(score)) for app, score in s.head(topk).items()]

class HybridWeighted:
    def __init__(self, cf_model: MF_Simple, cbf_model: CBF_TFIDF, alpha: float = 0.5):
        self.cf = cf_model; self.cbf = cbf_model; self.alpha = alpha

    def fit(self, train_df: pd.DataFrame, item_titles: pd.Series):
        pass  # 두 모델은 개별 fit 됨

    def recommend(self, user_id: str, candidate_items: List[str], topk: int = 10):
        scf = self.cf.score_series(user_id, candidate_items).reindex(candidate_items).fillna(0.0)
        scb = self.cbf.score_series(user_id, candidate_items).reindex(candidate_items).fillna(0.0)
        s = self.alpha * scf + (1.0 - self.alpha) * scb
        s = s.sort_values(ascending=False)
        return [(app, float(score)) for app, score in s.head(topk).items()]
