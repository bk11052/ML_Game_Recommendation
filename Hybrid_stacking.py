import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# (Stacking에 새로 필요한 라이브러리)
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
import random
import re
import warnings

# 경고 메시지 무시 (예: LogisticRegression 수렴 경고)
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 데이터 로드 및 전처리 (★핵심: 유령 유저 필터링 적용)
# ==============================================================================

print("1. 데이터 로드 및 전처리 시작...")

try:
    df_preprocessed = pd.read_csv('/Lab1/ML_T.P/preprocessed_data.csv')
    print("    -> 파일 로드 완료: preprocessed_data.csv")
except FileNotFoundError as e:
    print(f"오류: 'preprocessed_data.csv' 파일을 찾을 수 없습니다. ({e})")
    exit()

try:
    ratings_df = df_preprocessed[['author_id', 'app_id', 'is_positive_encoded', 'title']].copy()
    ratings_df.rename(columns={'is_positive_encoded': 'rating'}, inplace=True)
except KeyError:
    print("오류: preprocessed_data.csv에 필요한 컬럼이 없습니다.")
    exit()

ratings_df.dropna(subset=['author_id', 'app_id', 'rating'], inplace=True)
ratings_df['rating'] = ratings_df['rating'].astype(int)
ratings_df['title'] = ratings_df['title'].fillna('')

# (Stablility Fix) 최소 개 이상 리뷰를 남긴 유저만 필터링
print(f"    -> 원본 상호작용(리뷰) 수: {len(ratings_df)}")
user_counts = ratings_df['author_id'].value_counts()
active_users = user_counts[user_counts >= 1].index
ratings_df_final = ratings_df[ratings_df['author_id'].isin(active_users)].copy()

print(f"    -> (필터링 후) 상호작용 수: {len(ratings_df_final)}")
print(f"    -> (필터링 후) 유저 수: {len(active_users)}")

# ID 및 인덱스 매핑
user_to_index = {uid: i for i, uid in enumerate(ratings_df_final['author_id'].unique())}
game_to_index = {gid: i for i, gid in enumerate(ratings_df_final['app_id'].unique())}
index_to_game = {i: gid for gid, i in game_to_index.items()} 

ratings_df_final['u_idx'] = ratings_df_final['author_id'].map(user_to_index)
ratings_df_final['i_idx'] = ratings_df_final['app_id'].map(game_to_index)

n_users = len(user_to_index)
n_items = len(game_to_index)

if n_users == 0 or n_items == 0:
    print("오류: 2개 이상 리뷰를 남긴 유저가 없어 모델을 훈련할 수 없습니다.")
    exit()

R = csr_matrix((ratings_df_final['rating'].values, (ratings_df_final['u_idx'].values, ratings_df_final['i_idx'].values)),
               shape=(n_users, n_items))
print("1. 데이터 전처리 완료.")


# ==============================================================================
# 2. (Level 0) '전문가' 모델 훈련 (수업에서 배운 내용)
# ==============================================================================

# --- 2.A: CF '전문가' (MatrixFactorization) ---
class MatrixFactorization:
    def __init__(self, R, K, lr, reg, epochs):
        self.R = R
        self.n_users, self.n_items = R.shape
        self.K = K    
        self.lr = lr  
        self.reg = reg 
        self.epochs = epochs
        self.P = np.random.normal(scale=1./self.K, size=(self.n_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.n_items, self.K))
        
    def fit(self):
        print("    -> (Level 0) CF 모델 훈련 시작...")
        rows, cols = self.R.nonzero()
        ratings = self.R.data 
        for epoch in range(self.epochs):
            for u, i, r in zip(rows, cols, ratings):
                r_hat = self.predict(u, i)
                e = r - r_hat 
                self.P[u, :] += self.lr * (e * self.Q[i, :] - self.reg * self.P[u, :])
                self.Q[i, :] += self.lr * (e * self.P[u, :] - self.reg * self.Q[i, :])
        print("2.A: CF '전문가' 모델 훈련 완료.")

    def predict(self, u_idx, i_idx):
        return np.dot(self.P[u_idx, :], self.Q[i_idx, :])

    def predict_all(self, u_idx):
        return np.dot(self.P[u_idx, :], self.Q.T)

K_factors = 20
mf_model = MatrixFactorization(R, K=K_factors, lr=0.01, reg=0.01, epochs=20)
mf_model.fit()

# --- 2.B: CBF '전문가' (TF-IDF) ---
print("    -> (Level 0) CBF 모델 훈련 시작...")
game_id_map = pd.DataFrame(index_to_game.items(), columns=['i_idx', 'app_id'])
df_content = pd.merge(game_id_map, ratings_df_final[['app_id', 'title']].drop_duplicates(subset=['app_id']), 
                      on='app_id', how='left')
df_content = df_content.sort_values('i_idx').reset_index(drop=True)
df_content['title'] = df_content['title'].fillna('')

tfidf = TfidfVectorizer(token_pattern=r'\b\w{2,}\b')
tfidf_matrix = tfidf.fit_transform(df_content['title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# (CBF 예측 함수: 유저가 긍정 평가한 템과 i의 유사도 평균)
def get_cbf_score_for_user(user_id, cosine_sim_matrix):
    positive_ratings = ratings_df_final[(ratings_df_final['author_id'] == user_id) & (ratings_df_final['rating'] == 1)]
    rated_item_indices = positive_ratings['i_idx'].tolist()
    
    if not rated_item_indices:
        return pd.Series(0.0, index=range(n_items))
    
    # (n_items, n_rated_items)
    sim_scores = cosine_sim_matrix[:, rated_item_indices]
    
    # (n_items, ) - 각 아이템에 대해, 긍정 평가한 템과의 유사도 "평균"
    cbf_scores = sim_scores.mean(axis=1)
    
    return pd.Series(cbf_scores, index=range(n_items))

print("2.B: CBF '전문가' 모델 훈련 완료.")


# ==============================================================================
# 3. (Level 1) '매니저' 모델 훈련을 위한 "메타-데이터셋" 생성
# ==============================================================================
print("\n3. '매니저' 모델용 메타-데이터셋 생성 시작...")
print("    -> (주의: 유저 수가 많아 몇 분 정도 소요될 수 있습니다.)")

# 3.1: 모든 상호작용에 대해 CF '전문가'의 예측 점수 계산
print("    -> 3.A: CF 메타-피처 생성 중...")
# (u_idx, i_idx)
u_indices = ratings_df_final['u_idx'].values
i_indices = ratings_df_final['i_idx'].values
# (최적화) np.sum(P[u] * Q[i])
cf_meta_features = np.sum(mf_model.P[u_indices] * mf_model.Q[i_indices], axis=1)

# 3.2: 모든 상호작용에 대해 CBF '전문가'의 예측 점수 계산
print("    -> 3.B: CBF 메타-피처 생성 중...")
cbf_scores_by_user = {} # 유저별 CBF 예측 점수 캐싱
cbf_meta_features = []

for user_id, u_idx, i_idx in zip(ratings_df_final['author_id'], u_indices, i_indices):
    if user_id not in cbf_scores_by_user:
        # 이 유저의 모든 아이템에 대한 CBF 점수 목록을 미리 계산
        cbf_scores_by_user[user_id] = get_cbf_score_for_user(user_id, cosine_sim)
    
    # 해당 상호작용(i_idx)의 CBF 점수만 추출
    cbf_meta_features.append(cbf_scores_by_user[user_id].loc[i_idx])

# 3.3: 메타-데이터셋 (X_meta, y_meta) 완성
X_meta = pd.DataFrame({
    'cf_score': cf_meta_features,
    'cbf_score': cbf_meta_features
})
y_meta = ratings_df_final['rating'].values

print("3. 메타-데이터셋 생성 완료.")
print(f"    -> X_meta (입력) 크기: {X_meta.shape}")
print(f"    -> y_meta (정답) 크기: {y_meta.shape}")


# ==============================================================================
# 4. (Level 1) '매니저' 모델 훈련 (수업에서 안 배운 내용)
# ==============================================================================
print("\n4. (Level 1) '매니저' 모델(LogisticRegression) 훈련 시작...")

# (선택) 피처 스케일링: 두 '전문가'의 점수 범위를 0~1 사이로 맞춤
scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)

# '매니저' 모델(LogisticRegression) 훈련
stacker_model = LogisticRegression(random_state=42)
stacker_model.fit(X_meta_scaled, y_meta)

print("4. '매니저' 모델 훈련 완료.")

# (★핵심★) 학습된 가중치(중요도) 확인
# 0.3, 0.7 대신 모델이 '스스로' 학습한 가중치
learned_weights = stacker_model.coef_[0]
print("="*40)
print(f"**'매니저'가 학습한 가중치:**")
print(f"  -> CF 점수 가중치: {learned_weights[0]:.4f}")
print(f"  -> CBF 점수 가중치: {learned_weights[1]:.4f}")
print("="*40)


# ==============================================================================
# 5. Stacking 모델을 사용한 최종 추천
# ==============================================================================
print("\n5. Stacking 하이브리드 추천 결과 테스트...")

# Stacking 모델을 사용하는 추천 함수
def get_stacked_recommendation(user_id, n=5):
    if user_id not in user_to_index:
        return None, f"오류: 사용자 ID {user_id}는 훈련 데이터에 없습니다."
    
    u_idx = user_to_index[user_id]
    
    # 1. 모든 아이템(49개)에 대해 '전문가'들의 점수를 받음
    # (CF 전문가)
    S_CF = pd.Series(mf_model.predict_all(u_idx), index=range(n_items))
    # (CBF 전문가)
    if user_id not in cbf_scores_by_user: # 훈련 때 캐싱 안 된 유저일 경우
        S_CBF = get_cbf_score_for_user(user_id, cosine_sim)
    else:
        S_CBF = cbf_scores_by_user[user_id]
        
    # 2. '매니저' 모델이 사용할 메타-데이터셋 생성
    X_predict_meta = pd.DataFrame({
        'cf_score': S_CF,
        'cbf_score': S_CBF
    })
    
    # 3. '매니저' 모델이 최종 결정 (0.3/0.7 합산 대신)
    X_predict_meta_scaled = scaler.transform(X_predict_meta)
    
    # (★핵심★) 0~1 사이의 '긍정(1)' 확률을 최종 점수로 사용
    S_Hybrid = stacker_model.predict_proba(X_predict_meta_scaled)[:, 1]
    S_Hybrid = pd.Series(S_Hybrid, index=range(n_items))

    # 4. 이미 평가한 아이템 필터링
    rated_indices = ratings_df_final[ratings_df_final['author_id'] == user_id]['i_idx'].unique()
    S_Hybrid[rated_indices] = -np.inf
    
    top_indices = S_Hybrid.sort_values(ascending=False).head(n).index.tolist()
    
    # 5. 결과 반환
    game_title_map = df_content.set_index('app_id')['title'].to_dict()
    recommendation_list = []
    for i_idx in top_indices:
        gid = index_to_game[i_idx]
        recommendation_list.append({
            'title': game_title_map.get(gid, f"ID: {gid} (제목 없음)"),
            'app_id': gid,
            'hybrid_score': S_Hybrid.loc[i_idx],
        })
    return recommendation_list, None

# --- 테스트 실행 ---
test_user_id = random.choice(active_users)
recommendations, error_msg = get_stacked_recommendation(test_user_id, n=5)

rated_games_df = ratings_df_final[(ratings_df_final['author_id'] == test_user_id) & (ratings_df_final['rating'] == 1)]
rated_titles = rated_games_df['title'].drop_duplicates().tolist()

print("\n" + "="*80)
print(" **(Stacking 버전) 하이브리드 모델 추천 결과**")
print("="*80)
print(f"**테스트 사용자 ID**: {test_user_id}")
print(f"**이 유저가 긍정 평가한 게임 (취향)**: {', '.join(rated_titles)}")
print("-" * 40)
print("**이 유저를 위한 Top 5 추천 게임:**")

if error_msg:
    print(error_msg)
elif recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f" {i}. {rec['title']} (App ID: {rec['app_id']})")
        print(f"    (예측 확률: {rec['hybrid_score']:.4f})")
else:
    print("추천할 게임을 찾지 못했습니다.")
