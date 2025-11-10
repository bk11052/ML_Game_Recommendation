import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# ==============================================================================
# 모델명: CBF 단독 모델 (게임 제목 기반)
# ==============================================================================
# 코드 설명: 사용자의 긍정 평가 게임 '제목'을 기반으로 콘텐츠 유사도를 계산하여
#           가장 유사한 게임을 추천합니다. CF 모델은 완전히 제거되었습니다.

# 테스트 환경:
#   - 아이템 수: 50개 (극도로 희소한 데이터 환경)
#   - 테스트 유저 ID: 76561199200471638 (취향: Half-Life 시리즈)

# 주요 전략:
#   - CBF 100% 전략: "당신이 좋아한 게임과 제목이 가장 비슷한 게임"을
#     우선 추천하여 안정적인 고품질 추천을 생성합니다.
#   - 최종 점수 = CBF Score
# ==============================================================================


# ==============================================================================
# 1. 데이터 로드 및 전처리 (review.csv 사용, MIN_USER_INTERACTIONS=1)
# ==============================================================================

print("1. Data loading and preprocessing started...")

# 1.1. 파일 로드 (review.csv 사용)
try:
    # ⚠️ 인코딩 오류가 발생할 경우, 아래 encoding='cp949' 또는 'euc-kr'을 시도해보세요.
    df_reviews = pd.read_csv('review.csv', encoding='utf-8')
    df_games = pd.read_csv('games.csv', encoding='utf-8')
    print("   -> File loading complete: review.csv, games.csv")
except UnicodeDecodeError:
    try:
        # Fallback to CP949 for Windows compatibility if UTF-8 fails
        df_reviews = pd.read_csv('review.csv', encoding='cp949')
        df_games = pd.read_csv('games.csv', encoding='cp949')
        print("   -> File loading complete: review.csv, games.csv (using CP949 encoding)")
    except Exception as e:
        print(f"Error: Could not load files. Please check encoding and path. ({e})")
        exit()
except FileNotFoundError as e:
    print(f"Error: Files not found. Please check the path. ({e})")
    exit()

# 1.2. 평점 데이터 변환 및 필터링 (아이템 수 50개 확정 데이터셋)
MIN_USER_INTERACTIONS = 1 
df_reviews['rating'] = df_reviews['is_positive'].apply(lambda x: 1 if x == 'Positive' else 0)
ratings_df = df_reviews[['author_id', 'app_id', 'rating']].copy()
user_counts = ratings_df['author_id'].value_counts()
ratings_df_final = ratings_df[ratings_df['author_id'].isin(user_counts[user_counts >= MIN_USER_INTERACTIONS].index)].copy()

num_items_final = len(ratings_df_final['app_id'].unique())
num_users_final = len(ratings_df_final['author_id'].unique())

print(f"   -> [Filtering Result] Final User Count: {num_users_final}, Final Item Count: {num_items_final}")
if num_items_final != 50:
    print(f"Warning: Item count is {num_items_final}, not 50. This depends on the dataset.")
else:
    print("Item count confirmed at 50. Proceeding.")

# 1.3. ID 매핑 및 인덱스 준비 (CBF에 필요한 정보만 유지)
user_to_index = {uid: i for i, uid in enumerate(ratings_df_final['author_id'].unique())}
game_to_index = {gid: i for i, gid in enumerate(ratings_df_final['app_id'].unique())}
index_to_game = {i: gid for gid, i in game_to_index.items()} 

ratings_df_final['i_idx'] = ratings_df_final['app_id'].map(game_to_index)

n_items = len(game_to_index)

print(f"\n   --- Final Item Information ---")
print(f"   -> Final Item Count (n_items): {n_items}")
print("1. Data preprocessing complete.")


# ==============================================================================
# 2. CF 모델: Matrix Factorization (***제거됨***)
# ==============================================================================

# CF 모델 관련 클래스 및 훈련 코드는 CBF 100% 모델에서 모두 제거되었습니다.


# ==============================================================================
# 3. CBF 모델: TF-IDF 및 코사인 유사도 (게임 제목 기반)
# ==============================================================================

# 3.1. Content Data 준비 (게임 제목만 사용)
game_id_map = pd.DataFrame(index_to_game.items(), columns=['i_idx', 'app_id'])
df_content = pd.merge(game_id_map, df_games, on='app_id', how='left')
df_content = df_content.sort_values('i_idx').reset_index(drop=True)
df_content['content_text'] = df_content['title'].fillna('')

# 3.2. TF-IDF 벡터화 및 아이템 유사도 계산
tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b')
tfidf_matrix = tfidf.fit_transform(df_content['content_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3.3. CBF 점수 계산 함수 정의
def get_cbf_scores(user_id, all_item_indices, cosine_sim_matrix):
    """
    사용자의 긍정 평가 게임 제목을 기반으로 모든 아이템의 CBF 점수를 계산합니다.
    """
    
    positive_ratings = ratings_df_final[(ratings_df_final['author_id'] == user_id) & (ratings_df_final['rating'] == 1)]
    # 긍정 평가한 게임의 인덱스 목록
    rated_item_indices = [game_to_index[gid] for gid in positive_ratings['app_id'] if gid in game_to_index]

    if not rated_item_indices: 
        # 긍정 평가가 없으면 0점 반환
        return pd.Series(0.0, index=all_item_indices) 
        
    cbf_scores = np.zeros(n_items)
    
    # 긍정 평가 아이템들의 유사도 벡터 합산
    for i in all_item_indices:
        # i번째 아이템과 사용자가 긍정 평가한 모든 아이템 간의 유사도 합계
        cbf_scores[i] = np.sum(cosine_sim_matrix[i, rated_item_indices])
        
    # 점수 정규화 (최대 유사도 합계로 나누어 0~1 사이 값으로 만듦)
    max_sim_sum = np.max(cbf_scores)
    if max_sim_sum > 0: 
        cbf_scores /= max_sim_sum
        
    return pd.Series(cbf_scores, index=all_item_indices)


# ==============================================================================
# 4. 추천 함수 정의 (CBF 100% 단독)
# ==============================================================================

game_title_map = df_games[df_games['app_id'].isin(ratings_df_final['app_id'].unique())].set_index('app_id')['title']
all_item_indices = list(range(n_items))

def get_cbf_recommendation(user_id, n=10):
    """
    CBF 100% 단독 추천을 수행하는 함수.
    """
    if user_id not in ratings_df_final['author_id'].unique(): 
        return [f"Error: User ID {user_id} is not in the training data."]
    
    S_CBF = get_cbf_scores(user_id, all_item_indices, cosine_sim)

    # 이미 본 아이템 제외
    rated_indices = ratings_df_final[ratings_df_final['author_id'] == user_id]['i_idx'].unique()
    S_CBF = S_CBF[~S_CBF.index.isin(rated_indices)]
    
    # 상위 N개 추천
    top_indices = S_CBF.sort_values(ascending=False).head(n).index.tolist()
    
    cbf_recommendation_list = []
    for i_idx in top_indices:
        gid = index_to_game[i_idx]
        cbf_recommendation_list.append({
            'title': game_title_map.get(gid, f"ID: {gid} (Title Missing)"),
            'app_id': gid,
            'cbf_score': S_CBF.loc[i_idx],
        })
    return cbf_recommendation_list


# ==============================================================================
# 5. 최종 테스트 및 결과 출력
# ==============================================================================

print("\n5. CBF Standalone Recommendation Result Test...")

# 최소 1개 이상 긍정 평점을 남긴 유저를 랜덤으로 선택하여 테스트
valid_users = ratings_df_final[ratings_df_final['rating'] == 1]['author_id'].unique()
if len(valid_users) == 0:
    print("\nError: No user with positive ratings exists in the dataset to test CBF recommendation.")
    exit()

test_user_id = valid_users[random.randint(0, len(valid_users) - 1)]
N_REC = 5 

# ----------------------------------------------------------------------
# 5.1. CBF 단독 필터링 결과 출력 (English)
# ----------------------------------------------------------------------
cbf_recommendations = get_cbf_recommendation(test_user_id, n=N_REC)

rated_app_ids = ratings_df_final[ratings_df_final['author_id'] == test_user_id]['app_id'].unique()
rated_titles = df_games[df_games['app_id'].isin(rated_app_ids)]['title'].fillna('Title Missing').tolist()

print("\n" + "="*80)
print("**Content-Based Filtering (CBF) 100% Standalone Result (Title Based)**")
print("="*80)
print(f"**Test User ID**: {test_user_id}")
print(f"**Games Positively Rated by User (Preference)**: {', '.join(rated_titles)}")
print("-" * 40)
 
if isinstance(cbf_recommendations, list) and cbf_recommendations and 'cbf_score' in cbf_recommendations[0]:
    print(f"**Recommended Games List (Top {N_REC})**:")
    
    # 표 출력
    header = ["Rank", "Title", "CBF Score"]
    col_widths = [len(header[0]), 40, 15]

    for rec in cbf_recommendations:
        title = rec['title']
        if len(title) > 40:
            rec['title'] = title[:37] + '...'
        col_widths[1] = max(col_widths[1], len(rec['title']))

    format_str = f"| {{:<{col_widths[0]}}} | {{:<{col_widths[1]}}} | {{:>{col_widths[2]}}} |"

    print(format_str.format(header[0], header[1], header[2]))
    print("|" + "-" * (col_widths[0] + 2) + "|" + "-" * (col_widths[1] + 2) + "|" + "-" * (col_widths[2] + 2) + "|")

    for i, rec in enumerate(cbf_recommendations, 1):
        print(format_str.format(
            i,
            rec['title'],
            f"{rec['cbf_score']:.4f}"
        ))
    
    # 분석 메시지도 영어로 출력
    print("\nAnalysis: The recommendations are games whose titles have the most similar textual features, based on TF-IDF, to the games the user rated positively.")
else:
    if isinstance(cbf_recommendations, list):
        print(cbf_recommendations[0])
    else:
        print("An issue occurred while generating the recommendations.")
