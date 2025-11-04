import os, importlib
import pandas as pd
from collections import defaultdict

# =========================================================
# 1. 모델 어댑터 정의
# =========================================================
# - popularity (인기순)
# - hybrid.py (하이브리드 추천)
# - IBCF.py (아이템 기반 협업 필터링)
# =========================================================

def popularity_recommender(train_df: pd.DataFrame, topn=1000):
    """
    단순 인기순 추천 (Baseline)
    """
    pop = (train_df.groupby("app_id")["score"].sum()
           .sort_values(ascending=False).index.tolist())[:topn]
    user_hist = train_df.groupby("user_id")["app_id"].apply(set).to_dict()

    def rec(user_id, k):
        owned = user_hist.get(user_id, set())
        out = [i for i in pop if i not in owned]
        return out[:k]
    return rec

def _import_any(names):
    for nm in names:
        try:
            return importlib.import_module(nm)
        except Exception:
            pass
    return None

# =========================================================
# 2. 하이브리드 모델 로드
# =========================================================
def try_load_hybrid_adapter():
    m = _import_any(["hybrid","Hybrid"])
    if not m: return None
    if hasattr(m, "recommend"):
        return lambda user_id, k: list(m.recommend(user_id=user_id, k=k))
    if hasattr(m, "get_hybrid_recommendation"):
        def _call(uid, k):
            _, recs = m.get_hybrid_recommendation(uid, n=k)
            out=[]
            for r in recs:
                if isinstance(r, dict):
                    aid = r.get("app_id")
                    if aid is not None: out.append(str(aid))
                else:
                    out.append(str(r))
            return out[:k]
        return _call
    return None

# =========================================================
# 3. IBCF 모델 로드
# =========================================================
def try_load_ibcf_adapter():
    m = _import_any(["IBCF","ibcf"])
    if not m: return None
    if hasattr(m, "recommend"):
        return lambda user_id, k: list(m.recommend(user_id=user_id, k=k))
    return None

# =========================================================
# 4. 모델 딕셔너리 생성
# =========================================================
def build_model_dict(train_df: pd.DataFrame):
    models = {"popularity": popularity_recommender(train_df)}
    hb = try_load_hybrid_adapter()
    if hb: models["hybrid"] = hb
    ib = try_load_ibcf_adapter()
    if ib: models["ibcf"] = ib
    return models
