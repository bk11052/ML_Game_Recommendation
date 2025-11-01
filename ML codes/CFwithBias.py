# ==============================================================================
# 모델명: CF with Biases (편향이 추가된 행렬 분해)
# ==============================================================================
# 사용 목적 (Why):
#   - 기존 CF 모델은 유저의 '진짜 취향' 외에 '아이템의 일반적인 인기'나
#     '유저의 평균적인 평점 성향' 같은 일반 경향까지 학습합니다.
#   - 50개 아이템이라는 데이터 희소성 환경에서는 이 일반 경향(편향)이 노이즈가 됩니다.
#   - 이 편향을 분리하여, 나머지 잠재 요인이 순수한 취향 패턴만 학습하도록 안정성을 높이기 위해 사용했습니다.

# 사용 방법 (How):
#   - 기존 행렬 분해(MF) 예측 공식($P \cdot Q$)에 세 가지 편향 항을 추가했습니다.
#   - 예측 평점 = 전역 평균($\mu$) + 유저 편향($b_u$) + 아이템 편향($b_i$) + 잠재 요인($P \cdot Q$)
#   - 학습 시, 유저 편향($b_u$)과 아이템 편향($b_i$)을 $P, Q$와 함께 경사 하강법으로 동시에 업데이트합니다.

# 장단점:
#   - 장점: 기존 CF 코드에 비해 구현이 간단하면서도 CF 예측의 안정성과 정확도를 향상시킵니다.
#   - 단점: 여전히 콘텐츠 정보(CBF)의 강력한 보조 없이는 50개 아이템 환경의 근본적인 희소성을 완전히 해결하기 어렵습니다.
# ==============================================================================


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import random
import time

# ==============================================================================
# 1. 데이터 로드 및 전처리 (이전과 동일)
# ==============================================================================

# 파일 로드 (가정: review.csv와 games.csv는 접근 가능)
df_reviews = pd.read_csv('review.csv')
df_games = pd.read_csv('games.csv')

# 전처리 및 매핑 로직 재현
MIN_USER_INTERACTIONS = 1 
df_reviews['rating'] = df_reviews['is_positive'].apply(lambda x: 1 if x == 'Positive' else 0)
ratings_df = df_reviews[['author_id', 'app_id', 'rating']].copy()
user_counts = ratings_df['author_id'].value_counts()
ratings_df_final = ratings_df[ratings_df['author_id'].isin(user_counts[user_counts >= MIN_USER_INTERACTIONS].index)].copy()

user_to_index = {uid: i for i, uid in enumerate(ratings_df_final['author_id'].unique())}
game_to_index = {gid: i for i, gid in enumerate(ratings_df_final['app_id'].unique())}
index_to_game = {i: gid for gid, i in game_to_index.items()} 

ratings_df_final['u_idx'] = ratings_df_final['author_id'].map(user_to_index)
ratings_df_final['i_idx'] = ratings_df_final['app_id'].map(game_to_index)

n_users = len(user_to_index)
n_items = len(game_to_index)
R = csr_matrix((ratings_df_final['rating'].values, (ratings_df_final['u_idx'].values, ratings_df_final['i_idx'].values)),
               shape=(n_users, n_items))


# ==============================================================================
# 2. CF 모델: Matrix Factorization with Biases (편향 추가)
# ==============================================================================

class MatrixFactorizationWithBiases:
    def __init__(self, R, K, lr, reg, epochs):
        self.R = R
        self.n_users, self.n_items = R.shape
        self.K = K    
        self.lr = lr  
        self.reg = reg 
        self.epochs = epochs
        
        # 핵심 추가: 전역 평균(mu), 유저 편향(bu), 아이템 편향(bi)
        self.mu = self.R.data.mean()
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        
        # P와 Q 행렬 초기화 (잠재 요인)
        self.P = np.random.normal(scale=1./self.K, size=(self.n_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.n_items, self.K))
        
    def fit(self):
        rows, cols = self.R.nonzero()
        ratings = self.R.data 
        for epoch in range(self.epochs):
            for u, i, r in zip(rows, cols, ratings):
                
                # 예측 평점 공식에 편향 항 추가: mu + bu + bi + P*Q
                r_hat = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :])
                e = r - r_hat 
                
                # P, Q 업데이트 (기존 MF와 동일)
                self.P[u, :] += self.lr * (e * self.Q[i, :] - self.reg * self.P[u, :])
                self.Q[i, :] += self.lr * (e * self.P[u, :] - self.reg * self.Q[i, :])
                
                # 핵심 추가: 편향 항 업데이트 (규칙이 더 간단함)
                self.b_u[u] += self.lr * (e - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (e - self.reg * self.b_i[i])

    def predict_all(self, u_idx):
        # 예측 평점 공식에 편향 항 추가
        return self.mu + self.b_u[u_idx] + self.b_i + np.dot(self.P[u_idx, :], self.Q.T)

# 3. CF with Biases 모델 훈련
K_factors = 20
mf_bias_model = MatrixFactorizationWithBiases(R, K=K_factors, lr=0.01, reg=0.01, epochs=30)
mf_bias_model.fit()


# ==============================================================================
# 4. CF with Biases 추천 생성 및 결과 출력
# ==============================================================================

game_title_map = df_games[df_games['app_id'].isin(ratings_df_final['app_id'].unique())].set_index('app_id')['title']

def get_cf_recommendation(model, user_id, n=10):
    if user_id not in user_to_index: return None, [f"오류: 사용자 ID {user_id}는 훈련 데이터에 없습니다."]
    u_idx = user_to_index[user_id]
    
    predicted_ratings_all = model.predict_all(u_idx) 
    cf_scores = pd.Series(predicted_ratings_all, index=range(n_items))

    rated_indices = ratings_df_final[ratings_df_final['author_id'] == user_id]['i_idx'].unique()
    cf_scores = cf_scores[~cf_scores.index.isin(rated_indices)]
    
    top_indices = cf_scores.sort_values(ascending=False).head(n)
    
    recommendation_list = []
    for i_idx, score in top_indices.items():
        gid = index_to_game[i_idx]
        recommendation_list.append({
            'title': game_title_map.get(gid, f"ID: {gid} (제목 없음)"),
            'predicted_score': score,
        })
    return recommendation_list

# 테스트 유저 선정 (최소 2개 이상 리뷰 유저 중 랜덤 선택)
user_counts_filtered = user_counts[user_counts >= 2]
valid_users = ratings_df_final[ratings_df_final['author_id'].isin(user_counts_filtered.index)]['author_id'].unique()
test_user_id = valid_users[random.randint(0, len(valid_users) - 1)]
N_REC = 5 

cf_bias_recommendations = get_cf_recommendation(mf_bias_model, test_user_id, n=N_REC)

# 사용자가 긍정 평가한 게임 목록
rated_games_df = ratings_df_final[(ratings_df_final['author_id'] == test_user_id) & (ratings_df_final['rating'] == 1)]
rated_titles = df_games[df_games['app_id'].isin(rated_games_df['app_id'])]['title'].tolist()

print("\n" + "="*80)
print("**CF with Biases (편향 추가된 행렬 분해) 추천 결과**")
print("="*80)
print(f"**테스트 사용자 ID**: {test_user_id}")
print(f"**사용자가 긍정 평가한 게임 (취향)**: {', '.join(rated_titles)}")
print("-" * 40)
    
# 순수한 문자열 포맷팅으로 표 출력
header = ["순위", "제목", "CF Bias Score"]
cf_table_data = []
for i, rec in enumerate(cf_bias_recommendations, 1):
    cf_table_data.append([i, rec['title'], f"{rec['predicted_score']:.4f}"])

# 제목 길이에 맞춰 출력 포맷 조정
col_widths = [len(header[0]), 40, 15] 
for row in cf_table_data:
    title_len = len(row[1])
    if title_len > col_widths[1]:
        row[1] = row[1][:37] + '...'
        title_len = len(row[1])
    col_widths[1] = max(col_widths[1], title_len)

format_str = f"| {{:<{col_widths[0]}}} | {{:<{col_widths[1]}}} | {{:>{col_widths[2]}}} |"

print(format_str.format(header[0], header[1], header[2]))
print("|" + "-" * (col_widths[0] + 2) + "|" + "-" * (col_widths[1] + 2) + "|" + "-" * (col_widths[2] + 2) + "|")

for row in cf_table_data:
    print(format_str.format(row[0], row[1], row[2]))


print("\n💡 **분석:** 편향 항이 추가되어 아이템 인기도 영향이 분리되면서 순수한 취향 패턴 학습을 시도합니다.")
