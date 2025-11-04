import os, hashlib
import pandas as pd

# =========================================================
# 1. 데이터 스플릿 함수 정의
# =========================================================
# preprocessed_data.csv → user_id, app_id, score 기반으로
# ① 유저 홀드아웃(8:1:1) ② 시간기반(train/test) 분할을 수행합니다.
# =========================================================

def _col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def load_interactions(preprocessed_path="preprocessed_data.csv"):
    """
    전처리된 데이터를 불러와 user_id, app_id, score 정보를 추출합니다.
    """
    df = pd.read_csv(preprocessed_path)

    # 유저 ID 컬럼 탐색 (author_id, user_id 등)
    ucol = _col(df, ["author_id","user_id","username","profile","steamid"])
    if ucol is None:
        base = df[_col(df, ["id","content"])].astype(str)
        df["user_id"] = base.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:12])
    else:
        df = df.rename(columns={ucol:"user_id"})

    # 앱 ID / 평점 정보 매핑
    if "app_id" not in df.columns:
        raise ValueError("❌ app_id 컬럼이 필요합니다.")
    if "is_positive_encoded" in df.columns:
        df["score"] = df["is_positive_encoded"].fillna(0).astype(int)
    else:
        df["score"] = 1

    # 중복 제거 및 정제
    inter = df[["user_id","app_id","score"]].drop_duplicates()
    return inter, df


# =========================================================
# 2. 유저 홀드아웃 / 시간 분할 함수
# =========================================================
def user_holdout_split(inter, train=0.8, valid=0.1, seed=42):
    import numpy as np
    trs, vas, tes = [], [], []
    for uid, g in inter.groupby("user_id"):
        g = g.sample(frac=1.0, random_state=seed)
        n = len(g); nt = int(n*train); nv = int(n*valid)
        trs.append(g.iloc[:nt]); vas.append(g.iloc[nt:nt+nv]); tes.append(g.iloc[nt+nv:])
    return pd.concat(trs), pd.concat(vas), pd.concat(tes)


def time_split(df_raw, inter, train=0.8):
    """
    메모리 절약형 시간 분할:
    - df_raw에서 app_id별 대표 시간(최소/최초)을 1개로 요약
    - inter(app_id)에 map으로 매핑 (merge 대신 → 폭발 방지)
    """
    tcol = _col(df_raw, ["timestamp","date","date_release","created_at"])
    if tcol is None:
        # 시간 정보가 없으면 시간 분할은 생략
        return inter.copy(), inter.iloc[0:0].copy()

    # 1) df_raw에서 (app_id -> 대표시간) Series 만들기
    ts = (
        df_raw[["app_id", tcol]]
        .dropna()
        .assign(**{tcol: pd.to_datetime(df_raw[tcol], errors="coerce")})
        .dropna(subset=[tcol])
        .groupby("app_id")[tcol]
        .min()   # 앱별 '가장 이른' 시간 사용 (원하면 .max()로 바꿔도 됨)
    )

    # 2) inter에 시간 매핑 (dtype 정렬)
    inter2 = inter.copy()
    # app_id 타입을 맞춰야 map이 잘 맞습니다.
    try:
        inter2["app_id"] = inter2["app_id"].astype(ts.index.dtype)
    except Exception:
        inter2["app_id"] = inter2["app_id"].astype(str)
        ts.index = ts.index.astype(str)

    inter2["__ts__"] = inter2["app_id"].map(ts)

    # 3) 시간 없는 것 정리 (없으면 뒤로 밀리거나 학습으로 포함)
    # 시간 없는 데이터는 우선순위를 낮춰 뒤쪽으로 보내기 위해 큰 시간으로 채웁니다.
    fallback = pd.Timestamp.max
    inter2["__ts__"] = inter2["__ts__"].fillna(fallback)

    # 4) 시간 기준 정렬 → 앞쪽 train, 뒤쪽 test
    inter2 = inter2.sort_values("__ts__").drop(columns="__ts__")
    cut = int(len(inter2) * train)
    train_df = inter2.iloc[:cut][["user_id","app_id","score"]]
    test_df  = inter2.iloc[cut:][["user_id","app_id","score"]]
    return train_df, test_df



# =========================================================
# 3. 스플릿 실행 및 저장
# =========================================================
def make_splits(preprocessed_path="preprocessed_data.csv", outdir="evaluation/splits"):
    os.makedirs(outdir, exist_ok=True)
    inter, raw = load_interactions(preprocessed_path)

    tr, va, te = user_holdout_split(inter)
    tr.to_csv(f"{outdir}/train_user_holdout.csv", index=False)
    va.to_csv(f"{outdir}/valid_user_holdout.csv", index=False)
    te.to_csv(f"{outdir}/test_user_holdout.csv", index=False)

    tr_t, te_t = time_split(raw, inter)
    tr_t.to_csv(f"{outdir}/train_time.csv", index=False)
    te_t.to_csv(f"{outdir}/test_time.csv", index=False)

    print(f"✅ 스플릿 저장 완료 → {outdir}")

if __name__ == "__main__":
    make_splits()
