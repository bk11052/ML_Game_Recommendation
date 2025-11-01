# ==============================================================================
# 모델명: CBF 강화 하이브리드 모델 (콘텐츠 임베딩 시뮬레이션)
# ==============================================================================
# 사용 목적 (Why):
#   - 50개 아이템 환경에서 CF의 예측이 불안정하므로, 안정적인 CBF의 영향력을 극대화해야 합니다.
#   - 'developer', 'publisher' 컬럼이 없어 실제 임베딩(Word2Vec/BERT) 대신 콘텐츠 특징을 풍부하게 만듭니다.
#   - CBF에 70%의 높은 가중치를 부여하여, 유저가 좋아한 게임의 '맥락적 특징(title)'을 더 잘 파악하게 합니다.

# 사용 방법 (How):
#   - 유저가 긍정 평가한 게임의 'title'만 사용하여 TF-IDF 벡터를 생성하고, 코사인 유사도를 측정합니다.
#   - **Hybrid Score = (0.3 * CF Score) + (0.7 * CBF Score)** 공식을 사용해 최종 점수를 산출합니다.
#   - (주의: KeyError 수정으로 인해 현재는 title만 사용하며, 이는 초기 CBF와 동일한 결과입니다. 다른 콘텐츠 정보가 있다면 'content_text' 생성 시 추가해야 CBF가 강화됩니다.)

# 장단점:
#   - 장점: 구현이 쉽고, 데이터 희소성에 강하며, 유저의 명확한 취향(예: Half-Life 시리즈)을 확실하게 저격하여 추천 품질의 안정성을 보장합니다.
#   - 단점: CF와 CBF 점수를 나중에 섞는(Mixing) 방식이므로, 두 모델 간의 상호작용이 FM만큼 정교하지 못합니다. 유저가 선호하는 콘텐츠와 완전히 다른 새로운 게임을 추천하는 능력(세렌디피티)이 낮습니다.
# ==============================================================================


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re # 텍스트 정리용

# ==============================================================================
# 1. 데이터 로드 및 전처리
# ==============================================================================

print("1. 데이터 로드 및 전처리 시작...")

# 파일 로드 (가정: review.csv와 games.csv는 접근 가능)
try:
    df_reviews = pd.read_csv('review.csv')
    df_games = pd.read_csv('games.csv')
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
    exit()

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
print("1. 데이터 전처리 완료.")


# ==============================================================================
# 2. CF 모델: Matrix Factorization (표준 모델 사용)
# ==============================================================================

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
        rows, cols = self.R.nonzero()
        ratings = self.R.data 
        for epoch in range(self.epochs):
            for u, i, r in zip(rows, cols, ratings):
                r_hat = np.dot(self.P[u, :], self.Q[i, :])
                e = r - r_hat 
                self.P[u, :] += self.lr * (e * self.Q[i, :] - self.reg * self.P[u, :])
                self.Q[i, :] += self.lr * (e * self.P[u, :] - self.reg * self.Q[i, :])
        print("2. MF 모델 훈련 완료.")

    def predict_all(self, u_idx):
        return np.dot(self.P[u_idx, :], self.Q.T)

K_factors = 20
mf_model = MatrixFactorization(R, K=K_factors, lr=0.01, reg=0.01, epochs=30)
mf_model.fit()


# ==============================================================================
# 3. CBF 모델: 콘텐츠 유사도 계산 (오류 수정됨)
# ==============================================================================

# 3.1. 콘텐츠 데이터 준비
game_id_map = pd.DataFrame(index_to_game.items(), columns=['i_idx', 'app_id'])
df_content = pd.merge(game_id_map, df_games, on='app_id', how='left')
df_content = df_content.sort_values('i_idx').reset_index(drop=True)

def clean_text(text):
    if pd.isna(text): return ''
    text = str(text).lower().replace(' ', '')
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

# 핵심 수정: 'developer', 'publisher' 컬럼이 없어 발생한 KeyError를 회피.
#             현재는 'title' 컬럼만 사용하여 CBF 특징을 생성합니다.
df_content['content_text'] = df_content['title'].fillna('').apply(clean_text) 

# 3.2. TF-IDF 벡터화 및 아이템 유사도 계산
tfidf = TfidfVectorizer(token_pattern=r'\b\w{2,}\b')
tfidf_matrix = tfidf.fit_transform(df_content['content_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3.3. CBF 점수 계산 함수
def get_cbf_scores(user_id, all_item_indices, cosine_sim_matrix):
    positive_ratings = ratings_df_final[(ratings_df_final['author_id'] == user_id) & (ratings_df_final['rating'] == 1)]
    rated_item_indices = positive_ratings['i_idx'].tolist()
    if not rated_item_indices: return pd.Series(0.0, index=all_item_indices)
    cbf_scores = np.zeros(n_items)
    for i in all_item_indices:
        if i not in rated_item_indices:
            cbf_scores[i] = np.sum(cosine_sim_matrix[i, rated_item_indices])
    max_sim_sum = np.max(cbf_scores)
    if max_sim_sum > 0: cbf_scores /= max_sim_sum
    return pd.Series(cbf_scores, index=all_item_indices)


# ==============================================================================
# 4. 하이브리드 추천 함수 정의 (CBF 가중치 0.7)
# ==============================================================================

game_title_map = df_games[df_games['app_id'].isin(ratings_df_final['app_id'].unique())].set_index('app_id')['title']

def get_hybrid_recommendation(user_id, n=10, cf_weight=0.3, cbf_weight=0.7):
    if user_id not in user_to_index: return None, [f"오류: 사용자 ID {user_id}는 훈련 데이터에 없습니다."]
    u_idx = user_to_index[user_id]
    
    S_CF = pd.Series(mf_model.predict_all(u_idx), index=range(n_items))
    S_CBF = get_cbf_scores(user_id, range(n_items), cosine_sim)
    S_Hybrid = (cf_weight * S_CF) + (cbf_weight * S_CBF) 

    rated_indices = ratings_df_final[ratings_df_final['author_id'] == user_id]['i_idx'].unique()
    S_Hybrid = S_Hybrid[~S_Hybrid.index.isin(rated_indices)]
    S_CF_filtered = S_CF[~S_CF.index.isin(rated_indices)]
    S_CBF_filtered = S_CBF[~S_CBF.index.isin(rated_indices)]
    
    top_indices = S_Hybrid.sort_values(ascending=False).head(n).index.tolist()
    
    hybrid_recommendation_list = []
    for i_idx in top_indices:
        gid = index_to_game[i_idx]
        hybrid_recommendation_list.append({
            'title': game_title_map.get(gid, f"ID: {gid} (제목 없음)"),
            'cf_score': S_CF_filtered.loc[i_idx],
            'cbf_score': S_CBF_filtered.loc[i_idx],
            'hybrid_score': S_Hybrid.loc[i_idx],
        })
    return hybrid_recommendation_list


# ==============================================================================
# 5. 최종 테스트 및 결과 출력 (CBF 강화 버전)
# ==============================================================================

print("\n5. CBF 강화 (콘텐츠 임베딩 시뮬레이션) 하이브리드 추천 결과 테스트...")

user_counts_filtered = user_counts[user_counts >= 2]
valid_users = ratings_df_final[ratings_df_final['author_id'].isin(user_counts_filtered.index)]['author_id'].unique()
# 테스트 유저 랜덤 선택
test_user_id = valid_users[random.randint(0, len(valid_users) - 1)]
N_REC = 5 

hybrid_recommendations = get_hybrid_recommendation(test_user_id, n=N_REC, cf_weight=0.3, cbf_weight=0.7)

rated_games_df = ratings_df_final[(ratings_df_final['author_id'] == test_user_id) & (ratings_df_final['rating'] == 1)]
rated_titles = df_games[df_games['app_id'].isin(rated_games_df['app_id'])]['title'].tolist()

print("\n" + "="*80)
# 'CBF 강화'라는 표현은 유지하되, 현재는 Title만 사용했음을 인지해야 함
print("🎉 **CBF 강화 하이브리드 결과: (Title 기반 유사도 사용)**")
print("="*80)
print(f"**테스트 사용자 ID**: {test_user_id}")
print(f"**사용자가 긍정 평가한 게임 (취향)**: {', '.join(rated_titles)}")
print("-" * 40)
    
# 순수한 문자열 포맷팅으로 표 출력 (tabulate 미사용)
header = ["순위", "제목", "CF Score (0.3)", "CBF Score (0.7)", "최종 Hybrid Score"]
hybrid_table_data = []
for i, rec in enumerate(hybrid_recommendations, 1):
    hybrid_table_data.append([
        i, rec['title'], f"{rec['cf_score']:.4f}", f"{rec['cbf_score']:.4f}", f"**{rec['hybrid_score']:.4f}**"
    ])

col_widths = [len(header[0]), 40, 15, 15, 18] 
for row in hybrid_table_data:
    title_len = len(row[1])
    if title_len > col_widths[1]:
        row[1] = row[1][:37] + '...'
        title_len = len(row[1])
    col_widths[1] = max(col_widths[1], title_len)

format_str = f"| {{:<{col_widths[0]}}} | {{:<{col_widths[1]}}} | {{:^{col_widths[2]}}} | {{:^{col_widths[3]}}} | {{:>{col_widths[4]}}} |"

# 헤더 출력
print(format_str.format(header[0], header[1], header[2], header[3], header[4]))
print("|" + "-" * (col_widths[0] + 2) + "|" + "-" * (col_widths[1] + 2) + "|" + "-" * (col_widths[2] + 2) + "|" + "-" * (col_widths[3] + 2) + "|" + "-" * (col_widths[4] + 2) + "|")

# 데이터 출력
for row in hybrid_table_data:

    print(format_str.format(row[0], row[1], row[2], row[3], row[4]))
