import sys
import io
# 표준 출력(stdout) 인코딩을 UTF-8로 강제 설정
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
except AttributeError:
    pass


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import re, random

# ==============================================================================
# 1. 데이터 로드 및 전처리
# ==============================================================================

print("1. 데이터 로드 및 전처리 시작...")

# 1.1. 파일 로드 및 인코딩 지정 (글자 깨짐 방지)
try:
    df_reviews = pd.read_csv('review3.csv', encoding='utf-8') 
    df_games = pd.read_csv('games.csv', encoding='utf-8') 
    print("   -> 파일 로드 완료: review3.csv, games.csv") 
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e.filename})")
    print("경고: 파일이 없으므로 더미 데이터를 사용합니다.")
    
    # 더미 데이터 생성 (코드 구조 유지용)
    data_reviews = {
        'Labeling': ['Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive'],
        'username': ['UserA', 'UserA', 'UserB', 'UserB', 'UserC', 'UserC', 'UserA', 'UserB'],
        'game id': [100, 200, 100, 300, 200, 400, 500, 600]
    }
    data_games = {
        'app_id': [100, 200, 300, 400, 500, 600],
        'title': ['Adventure Game One', 'Space Shooter Pro', 'Medieval RPG Epic', 'Casual Puzzle Fun', 'Deep Strategy War', 'Simple Racing Sim']
    }
    df_reviews = pd.DataFrame(data_reviews)
    df_games = pd.DataFrame(data_games)


# 1.2. 컬럼 이름 변경 및 데이터 변환
df_reviews = df_reviews.rename(columns={'Labeling': 'is_positive', 
                                         'username': 'author_id', 
                                         'game id': 'app_id'})

df_reviews['rating'] = df_reviews['is_positive'].apply(lambda x: 1 if x == 'Positive' else 0)
ratings_df = df_reviews[['author_id', 'app_id', 'rating']].copy()

user_to_index = {uid: i for i, uid in enumerate(ratings_df['author_id'].unique())}
game_to_index = {gid: i for i, gid in enumerate(ratings_df['app_id'].unique())}

ratings_df['u_idx'] = ratings_df['author_id'].map(user_to_index)
ratings_df['i_idx'] = ratings_df['app_id'].map(game_to_index)

n_users, n_items = len(user_to_index), len(game_to_index)

R = csr_matrix(
    (ratings_df['rating'].values, (ratings_df['u_idx'].values, ratings_df['i_idx'].values)),
    shape=(n_users, n_items)
)
print("1. 데이터 전처리 완료.")


# ==============================================================================
# 2. CBF (Word2Vec) 모델 학습
# ==============================================================================

def clean_text(text):
    if pd.isna(text): return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

df_games['clean_title'] = df_games['title'].apply(clean_text)

# app_id가 game_to_index에 있는 게임만 사용
df_games_filtered = df_games[df_games['app_id'].isin(game_to_index)].copy() 

# --- Word2Vec 임베딩 학습 ---
tokenized_titles = [t.split() for t in df_games_filtered['clean_title']]

print("2. Word2Vec 학습 중...")
w2v_model = Word2Vec(sentences=tokenized_titles, vector_size=30, window=3, min_count=1, sg=1, workers=4)

def get_w2v_vector(tokens):
    """단어 리스트의 평균 Word2Vec 벡터 계산"""
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

print("   -> Word2Vec 벡터 계산 중...")
w2v_vectors = np.vstack([get_w2v_vector(tokens) for tokens in tokenized_titles])
w2v_sim = cosine_similarity(w2v_vectors)

# --- K-Means 클러스터링 ---
NUM_CLUSTERS = min(5, len(df_games_filtered))
print("   -> K-Means 클러스터링 수행 중...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
df_games_filtered['i_idx'] = df_games_filtered['app_id'].map(game_to_index)
df_games_filtered['cluster'] = kmeans.fit_predict(w2v_vectors)
print("2. Word2Vec 기반 CBF 모델 준비 완료.")


# ==============================================================================
# 3. Word2Vec 단독 추천 함수 정의 (Fallback 로직 강화)
# ==============================================================================
def get_w2v_recommendation(user_id, n=5):
    # ... (생략: user_id 체크 및 user_rated_i_idx 계산 부분은 동일) ...
    
    if user_id not in user_to_index:
        return pd.DataFrame()
    
    user_rated_apps = ratings_df[(ratings_df['author_id'] == user_id) & (ratings_df['rating'] == 1)]['app_id'].tolist()
    user_rated_i_idx = [game_to_index[app_id] for app_id in user_rated_apps if app_id in game_to_index]

    # 긍정 평가가 없는 경우 빈 DataFrame 반환
    if not user_rated_i_idx:
        return pd.DataFrame()

    # i_idx가 df_games_filtered에 있는 항목에 대한 매핑
    game_idx_map = {idx: i for i, idx in enumerate(df_games_filtered['i_idx'].tolist())}
    
    # 미평가된 아이템 (i_idx) 목록을 먼저 계산
    unrated_i_idx = [i for i in range(n_items) if i not in user_rated_i_idx]
    
    if not unrated_i_idx:
         print(f"\n[Error] 유저 {user_id}는 모든 게임을 평가했습니다. 추천할 아이템이 없습니다.")
         return pd.DataFrame() # 평가할 아이템이 없다면 빈 목록 반환

    # ----------------------------------------------------
    # Word2Vec 유사도 기반 CBF 점수 계산
    # ----------------------------------------------------
    S_CBF_W2V = np.zeros(n_items)
    for i_idx in unrated_i_idx: # 미평가된 아이템만 대상으로 계산
        if i_idx in game_idx_map:
            current_idx = game_idx_map[i_idx]
            rated_indices_in_filtered = [game_idx_map[idx] for idx in user_rated_i_idx if idx in game_idx_map]
            
            if rated_indices_in_filtered:
                S_CBF_W2V[i_idx] = np.sum(w2v_sim[current_idx, rated_indices_in_filtered])

    S_W2V_Score = pd.Series(S_CBF_W2V, index=range(n_items))
    
    # 정규화
    max_w2v_score = S_W2V_Score.max()
    S_W2V_Score /= max_w2v_score if max_w2v_score > 0 else 1.0 
    
    # 평가했던 항목 제외 (이 시점에서는 S_W2V_Score에 미평가 항목만 남아있음)
    S_W2V_Score = S_W2V_Score[~S_W2V_Score.index.isin(user_rated_i_idx)]

    # Word2Vec 점수가 모두 0인지 확인
    if S_W2V_Score.sum() == 0:
        print(f"\n[Fallback] Word2Vec 유사도가 0이므로, 미평가된 아이템 중 랜덤 아이템을 추천합니다.")
        
        # 미평가된 아이템 중 랜덤 선택
        fallback_indices = random.sample(unrated_i_idx, min(n, len(unrated_i_idx)))
        
        df_games_sorted = df_games_filtered.set_index('i_idx').sort_index()
        # Fallback은 0점을 가지므로, Indexing 시 문제가 발생할 수 있어 loc 사용
        recs = df_games_sorted.loc[fallback_indices, ['title', 'cluster']].copy()
        recs['W2V CBF Score'] = 0.0 
        recs.columns = ['Title', 'Cluster', 'W2V CBF Score (Fallback)']
        return recs
        
    # CBF 점수 기반 추천 (점수 0이 아닌 경우)
    top_indices = S_W2V_Score.sort_values(ascending=False).head(n).index.tolist()
    
    df_games_sorted = df_games_filtered.set_index('i_idx').sort_index()
    
    recs = df_games_sorted.loc[top_indices, ['title', 'cluster']].copy()
    
    recs['W2V CBF Score'] = S_W2V_Score[top_indices].values
    
    recs.columns = ['Title', 'Cluster', 'W2V CBF Score']
    return recs

# ==============================================================================
# 4. 테스트 실행 (Word2Vec 단독) - 유저 고정
# ==============================================================================

# 테스트 유저 ID를 요청하신 값으로 고정
TEST_USER_ID = '76561197996337391'

# 고정된 유저 ID가 데이터에 있는지 확인
if TEST_USER_ID not in user_to_index:
    print(f"경고: 요청된 유저 ID {TEST_USER_ID}가 훈련 데이터에 존재하지 않아, 랜덤 유저를 선택합니다.")
    
    # 긍정 평가를 1개 이상 남긴 사용자 목록 필터링
    users_with_positive_ratings = ratings_df[ratings_df['rating'] == 1]['author_id'].unique()
    
    if users_with_positive_ratings.size == 0:
        test_user = 'UserA'
    else:
        test_user = random.choice(users_with_positive_ratings)
else:
    test_user = TEST_USER_ID

# (주의: 만약 이 유저가 모든 게임을 평가했다면, 추천이 나오지 않을 수 있습니다.)

print("\n4. 추천 결과 (Word2Vec 기반 CBF 단독 추천)\n")
recs = get_w2v_recommendation(test_user, n=5)

print(f"Test User ID: {test_user}\n")
print("Top 5 Word2Vec Recommendations:")
print("------------------------------")
if not recs.empty:
    # 인덱스(순위)를 1부터 시작하도록 출력
    recs.index = range(1, len(recs) + 1)
    
    # 사용자가 긍정 평가한 게임 목록 출력 (취향 확인)
    rated_titles = df_reviews[
        (df_reviews['author_id'] == test_user) & (df_reviews['rating'] == 1)
    ]['app_id'].map(df_games.set_index('app_id')['title']).dropna().tolist()
    
    print(f"User's Positive Rated Games (Taste): {', '.join(rated_titles[:3])}{'...' if len(rated_titles) > 3 else ''}")
    print("-" * 30)
    
    # 컬럼명이 Fallback인지 아닌지에 따라 to_string 옵션 조정
    float_format_str = "%.4f" if 'Fallback' not in recs.columns[2] else "%s" 
    print(recs.to_string(float_format=float_format_str))
else:
    print("No recommendations were possible (The user may have already rated all available items).")import sys
import io
# 표준 출력(stdout) 인코딩을 UTF-8로 강제 설정
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
except AttributeError:
    pass


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import re, random

# ==============================================================================
# 1. 데이터 로드 및 전처리
# ==============================================================================

print("1. 데이터 로드 및 전처리 시작...")

# 1.1. 파일 로드 및 인코딩 지정 (글자 깨짐 방지)
try:
    df_reviews = pd.read_csv('review3.csv', encoding='utf-8') 
    df_games = pd.read_csv('games.csv', encoding='utf-8') 
    print("   -> 파일 로드 완료: review3.csv, games.csv") 
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e.filename})")
    print("경고: 파일이 없으므로 더미 데이터를 사용합니다.")
    
    # 더미 데이터 생성 (코드 구조 유지용)
    data_reviews = {
        'Labeling': ['Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive'],
        'username': ['UserA', 'UserA', 'UserB', 'UserB', 'UserC', 'UserC', 'UserA', 'UserB'],
        'game id': [100, 200, 100, 300, 200, 400, 500, 600]
    }
    data_games = {
        'app_id': [100, 200, 300, 400, 500, 600],
        'title': ['Adventure Game One', 'Space Shooter Pro', 'Medieval RPG Epic', 'Casual Puzzle Fun', 'Deep Strategy War', 'Simple Racing Sim']
    }
    df_reviews = pd.DataFrame(data_reviews)
    df_games = pd.DataFrame(data_games)


# 1.2. 컬럼 이름 변경 및 데이터 변환
df_reviews = df_reviews.rename(columns={'Labeling': 'is_positive', 
                                         'username': 'author_id', 
                                         'game id': 'app_id'})

df_reviews['rating'] = df_reviews['is_positive'].apply(lambda x: 1 if x == 'Positive' else 0)
ratings_df = df_reviews[['author_id', 'app_id', 'rating']].copy()

user_to_index = {uid: i for i, uid in enumerate(ratings_df['author_id'].unique())}
game_to_index = {gid: i for i, gid in enumerate(ratings_df['app_id'].unique())}

ratings_df['u_idx'] = ratings_df['author_id'].map(user_to_index)
ratings_df['i_idx'] = ratings_df['app_id'].map(game_to_index)

n_users, n_items = len(user_to_index), len(game_to_index)

R = csr_matrix(
    (ratings_df['rating'].values, (ratings_df['u_idx'].values, ratings_df['i_idx'].values)),
    shape=(n_users, n_items)
)
print("1. 데이터 전처리 완료.")


# ==============================================================================
# 2. CBF (Word2Vec) 모델 학습
# ==============================================================================

def clean_text(text):
    if pd.isna(text): return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

df_games['clean_title'] = df_games['title'].apply(clean_text)

# app_id가 game_to_index에 있는 게임만 사용
df_games_filtered = df_games[df_games['app_id'].isin(game_to_index)].copy() 

# --- Word2Vec 임베딩 학습 ---
tokenized_titles = [t.split() for t in df_games_filtered['clean_title']]

print("2. Word2Vec 학습 중...")
w2v_model = Word2Vec(sentences=tokenized_titles, vector_size=30, window=3, min_count=1, sg=1, workers=4)

def get_w2v_vector(tokens):
    """단어 리스트의 평균 Word2Vec 벡터 계산"""
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

print("   -> Word2Vec 벡터 계산 중...")
w2v_vectors = np.vstack([get_w2v_vector(tokens) for tokens in tokenized_titles])
w2v_sim = cosine_similarity(w2v_vectors)

# --- K-Means 클러스터링 ---
NUM_CLUSTERS = min(5, len(df_games_filtered))
print("   -> K-Means 클러스터링 수행 중...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
df_games_filtered['i_idx'] = df_games_filtered['app_id'].map(game_to_index)
df_games_filtered['cluster'] = kmeans.fit_predict(w2v_vectors)
print("2. Word2Vec 기반 CBF 모델 준비 완료.")


# ==============================================================================
# 3. Word2Vec 단독 추천 함수 정의 (Fallback 로직 강화)
# ==============================================================================
def get_w2v_recommendation(user_id, n=5):
    # ... (생략: user_id 체크 및 user_rated_i_idx 계산 부분은 동일) ...
    
    if user_id not in user_to_index:
        return pd.DataFrame()
    
    user_rated_apps = ratings_df[(ratings_df['author_id'] == user_id) & (ratings_df['rating'] == 1)]['app_id'].tolist()
    user_rated_i_idx = [game_to_index[app_id] for app_id in user_rated_apps if app_id in game_to_index]

    # 긍정 평가가 없는 경우 빈 DataFrame 반환
    if not user_rated_i_idx:
        return pd.DataFrame()

    # i_idx가 df_games_filtered에 있는 항목에 대한 매핑
    game_idx_map = {idx: i for i, idx in enumerate(df_games_filtered['i_idx'].tolist())}
    
    # 미평가된 아이템 (i_idx) 목록을 먼저 계산
    unrated_i_idx = [i for i in range(n_items) if i not in user_rated_i_idx]
    
    if not unrated_i_idx:
         print(f"\n[Error] 유저 {user_id}는 모든 게임을 평가했습니다. 추천할 아이템이 없습니다.")
         return pd.DataFrame() # 평가할 아이템이 없다면 빈 목록 반환

    # ----------------------------------------------------
    # Word2Vec 유사도 기반 CBF 점수 계산
    # ----------------------------------------------------
    S_CBF_W2V = np.zeros(n_items)
    for i_idx in unrated_i_idx: # 미평가된 아이템만 대상으로 계산
        if i_idx in game_idx_map:
            current_idx = game_idx_map[i_idx]
            rated_indices_in_filtered = [game_idx_map[idx] for idx in user_rated_i_idx if idx in game_idx_map]
            
            if rated_indices_in_filtered:
                S_CBF_W2V[i_idx] = np.sum(w2v_sim[current_idx, rated_indices_in_filtered])

    S_W2V_Score = pd.Series(S_CBF_W2V, index=range(n_items))
    
    # 정규화
    max_w2v_score = S_W2V_Score.max()
    S_W2V_Score /= max_w2v_score if max_w2v_score > 0 else 1.0 
    
    # 평가했던 항목 제외 (이 시점에서는 S_W2V_Score에 미평가 항목만 남아있음)
    S_W2V_Score = S_W2V_Score[~S_W2V_Score.index.isin(user_rated_i_idx)]

    # Word2Vec 점수가 모두 0인지 확인
    if S_W2V_Score.sum() == 0:
        print(f"\n[Fallback] Word2Vec 유사도가 0이므로, 미평가된 아이템 중 랜덤 아이템을 추천합니다.")
        
        # 미평가된 아이템 중 랜덤 선택
        fallback_indices = random.sample(unrated_i_idx, min(n, len(unrated_i_idx)))
        
        df_games_sorted = df_games_filtered.set_index('i_idx').sort_index()
        # Fallback은 0점을 가지므로, Indexing 시 문제가 발생할 수 있어 loc 사용
        recs = df_games_sorted.loc[fallback_indices, ['title', 'cluster']].copy()
        recs['W2V CBF Score'] = 0.0 
        recs.columns = ['Title', 'Cluster', 'W2V CBF Score (Fallback)']
        return recs
        
    # CBF 점수 기반 추천 (점수 0이 아닌 경우)
    top_indices = S_W2V_Score.sort_values(ascending=False).head(n).index.tolist()
    
    df_games_sorted = df_games_filtered.set_index('i_idx').sort_index()
    
    recs = df_games_sorted.loc[top_indices, ['title', 'cluster']].copy()
    
    recs['W2V CBF Score'] = S_W2V_Score[top_indices].values
    
    recs.columns = ['Title', 'Cluster', 'W2V CBF Score']
    return recs

# ==============================================================================
# 4. 테스트 실행 (Word2Vec 단독) - 유저 고정
# ==============================================================================

# 테스트 유저 ID를 요청하신 값으로 고정
TEST_USER_ID = '76561197996337391'

# 고정된 유저 ID가 데이터에 있는지 확인
if TEST_USER_ID not in user_to_index:
    print(f"경고: 요청된 유저 ID {TEST_USER_ID}가 훈련 데이터에 존재하지 않아, 랜덤 유저를 선택합니다.")
    
    # 긍정 평가를 1개 이상 남긴 사용자 목록 필터링
    users_with_positive_ratings = ratings_df[ratings_df['rating'] == 1]['author_id'].unique()
    
    if users_with_positive_ratings.size == 0:
        test_user = 'UserA'
    else:
        test_user = random.choice(users_with_positive_ratings)
else:
    test_user = TEST_USER_ID

# (주의: 만약 이 유저가 모든 게임을 평가했다면, 추천이 나오지 않을 수 있습니다.)

print("\n4. 추천 결과 (Word2Vec 기반 CBF 단독 추천)\n")
recs = get_w2v_recommendation(test_user, n=5)

print(f"Test User ID: {test_user}\n")
print("Top 5 Word2Vec Recommendations:")
print("------------------------------")
if not recs.empty:
    # 인덱스(순위)를 1부터 시작하도록 출력
    recs.index = range(1, len(recs) + 1)
    
    # 사용자가 긍정 평가한 게임 목록 출력 (취향 확인)
    rated_titles = df_reviews[
        (df_reviews['author_id'] == test_user) & (df_reviews['rating'] == 1)
    ]['app_id'].map(df_games.set_index('app_id')['title']).dropna().tolist()
    
    print(f"User's Positive Rated Games (Taste): {', '.join(rated_titles[:3])}{'...' if len(rated_titles) > 3 else ''}")
    print("-" * 30)
    
    # 컬럼명이 Fallback인지 아닌지에 따라 to_string 옵션 조정
    float_format_str = "%.4f" if 'Fallback' not in recs.columns[2] else "%s" 
    print(recs.to_string(float_format=float_format_str))
else:
    print("No recommendations were possible (The user may have already rated all available items).")
