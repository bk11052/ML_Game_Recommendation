import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from scipy.sparse import hstack, csr_matrix

# =========================================================
# 1️⃣ 데이터 로드
# =========================================================
try:
    users = pd.read_csv('C:/Lab1/ML_T.P/users.csv')
    games_data = pd.read_csv('C:/Lab1/ML_T.P/games.csv')
    recommendation_data = pd.read_csv('C:/Lab1/ML_T.P/recommendations.csv')
except FileNotFoundError as e:
    print(f"❌ 파일 로드 실패: {e}")
    exit()

# =========================================================
# 2️⃣ 협업 필터링용 전처리
# =========================================================
merged_data = games_data.merge(recommendation_data, on='app_id')
merged_data.drop(columns=['helpful', 'funny', 'date', 'hours', 'review_id'], inplace=True)

# 활동량 기준 필터링
user_count = merged_data['user_id'].value_counts()
game_count = merged_data['title'].value_counts()

filtered_data = merged_data[
    merged_data['user_id'].isin(user_count[user_count >= 500].index) &
    merged_data['title'].isin(game_count[game_count >= 350].index)
]

# 피벗 테이블 생성 (게임 x 사용자)
pt = filtered_data.pivot_table(
    index='title', columns='user_id', values='is_recommended', fill_value=0
)

# 코사인 유사도로 CF 유사도 계산
cf_similarity_matrix = cosine_similarity(pt)
cf_similarity_df = pd.DataFrame(cf_similarity_matrix, index=pt.index, columns=pt.index)

print(f"✅ CF 유사도 행렬 크기: {cf_similarity_df.shape}")

# =========================================================
# 3️⃣ 콘텐츠 기반 필터링용 전처리
# =========================================================
# 영어 게임만 추출
games_data['title'] = games_data['title'].fillna('')
games_data['date_release'] = pd.to_datetime(games_data['date_release'], errors='coerce')
games_data = games_data.dropna(subset=['title', 'positive_ratio', 'price_final', 'rating'])

games_data['release_year'] = games_data['date_release'].dt.year.fillna(0)

# 텍스트(타이틀) → 벡터
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(games_data['title'])

# 수치형 스케일링
scaler = MinMaxScaler()
games_data[['positive_ratio', 'price_final', 'release_year']] = scaler.fit_transform(
    games_data[['positive_ratio', 'price_final', 'release_year']]
)
scaled_features_sparse = csr_matrix(games_data[['positive_ratio', 'price_final', 'release_year']])

# 평점(rating) 인코딩
ratings_categories = [
    'Overwhelmingly Negative', 'Very Negative', 'Mostly Negative', 'Negative',
    'Mixed', 'Positive', 'Mostly Positive', 'Very Positive', 'Overwhelmingly Positive'
]
oe = OrdinalEncoder(categories=[ratings_categories])
games_data['rating'] = oe.fit_transform(games_data[['rating']])

# OS 지원 여부 이진 변환
for col in ['win', 'mac', 'linux']:
    if col in games_data.columns:
        games_data[col] = games_data[col].astype(int)
    else:
        games_data[col] = 0

binary_features_matrix = csr_matrix(games_data[['win', 'mac', 'linux']])

# 전체 콘텐츠 특징 결합
combined_features = hstack([
    vector,
    csr_matrix(games_data[['rating']]),
    scaled_features_sparse,
    binary_features_matrix
])

# 콘텐츠 기반 유사도 계산
content_similarity_matrix = cosine_similarity(combined_features)
content_similarity_df = pd.DataFrame(content_similarity_matrix, index=games_data['title'], columns=games_data['title'])

print(f"✅ 콘텐츠 유사도 행렬 크기: {content_similarity_df.shape}")

# =========================================================
# 4️⃣ 하이브리드 유사도 결합 (IBCF 비중 크게 설정정)
# =========================================================
CONTENT_WEIGHT = 0.3
CF_WEIGHT = 0.7

# 교집합(둘 다 존재하는 게임만)으로 정렬
common_titles = list(set(pt.index) & set(games_data['title']))
cf_sim_sub = cf_similarity_df.loc[common_titles, common_titles]
content_sim_sub = content_similarity_df.loc[common_titles, common_titles]

# 하이브리드 유사도 계산
hybrid_similarity = (CF_WEIGHT * cf_sim_sub) + (CONTENT_WEIGHT * content_sim_sub)
print(f"✅ 하이브리드 유사도 행렬 크기: {hybrid_similarity.shape}")

# =========================================================
# 5️⃣ 추천 함수 정의
# =========================================================
def recommend_hybrid(game_title, n=5):
    if game_title not in hybrid_similarity.index:
        return [f"❌ '{game_title}'은(는) 데이터셋에 존재하지 않습니다."]
    sim_scores = hybrid_similarity.loc[game_title].sort_values(ascending=False)
    recommendations = sim_scores[sim_scores.index != game_title].head(n)
    return recommendations.index.tolist()

# =========================================================
# 6️⃣ 테스트
# =========================================================
test_game = "Teenage Mutant Ninja Turtles: Shredder's Revenge"
recommendations = recommend_hybrid(test_game, n=5)

print("\n=================================================================")
print(f"🔥 하이브리드 추천 결과 (입력 게임: {test_game})")
print("-----------------------------------------------------------------")
for i, title in enumerate(recommendations, start=1):
    print(f"  {i}. {title}")
print("=================================================================")
