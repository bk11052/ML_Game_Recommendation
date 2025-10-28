import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 1. 데이터 로드 및 전처리  ---
try:
    users = pd.read_csv('C:/Lab1/ML_T.P/users.csv')
    games_data = pd.read_csv('C:/Lab1/ML_T.P/games.csv')
    recommendation_data = pd.read_csv('C:/Lab1/ML_T.P/recommendations.csv')
except FileNotFoundError as e:
    print(f"❌ 파일 로드 실패: {e}. 경로를 확인해주세요.")
    exit()

# 데이터 결합 및 클리닝
merged_data = games_data.merge(recommendation_data, on='app_id')
merged_data.drop(columns=['helpful', 'funny', 'date', 'hours', 'review_id'], inplace=True)

# 활동량 기준 필터링
user_count = merged_data['user_id'].value_counts()
game_count = merged_data['title'].value_counts()
filtered_data = merged_data[
    merged_data['user_id'].isin(user_count[user_count >= 500].index) & 
    merged_data['title'].isin(game_count[game_count >= 350].index)
]

# 피벗 테이블 생성 (게임(인덱스) x 사용자(컬럼) 행렬)
pt = filtered_data.pivot_table(
    index='title', 
    columns='user_id', 
    values='is_recommended', 
    fill_value=0
)
print(f"✅ 최종 행렬 크기 (게임 수 x 사용자 수): {pt.shape}")

# ==============================================================================
# 2. 아이템 기반 CF (IBCF) 구현을 위한 유사도 계산
# ==============================================================================

# 게임-사용자 행렬 (pt)을 사용하여 게임 간 코사인 유사도를 계산
# pt의 각 행은 한 게임에 대한 모든 사용자의 피드백 벡터
similarity_matrix = cosine_similarity(pt)
print(f"✅ 게임 간 유사도 행렬 크기: {similarity_matrix.shape}")


# ==============================================================================
# 3. 사진과 동일한 함수 정의 (아이템 기반 추천 함수)
# ==============================================================================

def recommend_based_on_collaborative(game, n_suggestions=5):
    """
    입력된 게임 제목을 기반으로 유사한 게임을 찾아 추천합니다. (IBCF)
    
    Args:
        game (str): 기준이 될 게임의 제목 (pt의 인덱스에 존재해야 함).
        n_suggestions (int): 추천할 게임의 개수.

    Returns:
        list: 추천 게임 제목 리스트.
    """
    
    # 1. 입력 게임의 인덱스 찾기
    try:
        index = np.where(pt.index == game)[0][0]
    except IndexError:
        return [f"Error: Game '{game}' not found in the filtered list."]
    
    # 2. 해당 게임의 유사도 벡터를 가져와 유사도 순으로 정렬
    # (인덱스, 유사도 점수) 튜플 리스트를 생성하고 유사도 점수(x[1])를 기준으로 내림차순 정렬
    sim_games = sorted(
        list(enumerate(similarity_matrix[index])), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # 3. 가장 유사한 Top N+1개 항목을 선택하고 (첫 번째는 자기 자신이므로 제외)
    # 1부터 N+1까지 슬라이싱하여 Top N개 추천
    suggestions = []
    
    # 슬라이싱 [1: n_suggestions + 1]을 사용하여 자기 자신을 제외하고 상위 N개를 선택
    top_n_similar = sim_games[1: n_suggestions + 1] 
    
    # 4. 인덱스를 실제 게임 제목으로 변환하여 리스트에 추가
    for item in top_n_similar:
        game_index = item[0]
        suggestions.append(pt.index[game_index])
        
    return suggestions

# ==============================================================================
# 4. 추천 실행 예시
# ==============================================================================

# 게임 이름 직접 입력
test_game_title = "Teenage Mutant Ninja Turtles: Shredder's Revenge"

# 추천 함수 실행
recommended_list = recommend_based_on_collaborative(test_game_title, n_suggestions=5)

print("\n=================================================================")
print(f"🔥 아이템 기반 CF 추천 결과 (입력 게임: {test_game_title})")
print("-----------------------------------------------------------------")
print("💡 추천 게임 목록:")
for i, title in enumerate(recommended_list):
    print(f"  {i+1}. {title}")
print("=================================================================")
