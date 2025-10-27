import numpy as np 
import pandas as pd 
import random
from sklearn.model_selection import train_test_split

# --- 1. 데이터 로드 및 전처리 (제시된 코드를 통합) ---
try:
    # NOTE: 파일 경로는 실제 환경에 맞게 수정이 필요합니다.
    users = pd.read_csv('C:/Lab1/ML_T.P/users.csv')
    games_data = pd.read_csv('C:/Lab1/ML_T.P/games.csv')
    recommendation_data = pd.read_csv('C:/Lab1/ML_T.P/recommendations.csv')
    print("✅ 데이터 로드 완료.")
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
print(f"✅ 필터링 후 데이터 크기: {filtered_data.shape[0]} 행")

# 피벗 테이블 생성 (0/1 이진 평점: 1=추천, 0=미추천/무관심)
pt = filtered_data.pivot_table(
    index='title', 
    columns='user_id', 
    values='is_recommended', # 이진 값 (True=1, False=0)
    fill_value=0
)
print(f"✅ 최종 행렬 크기 (게임 수 x 사용자 수): {pt.shape}")

# 학습을 위한 데이터프레임 변환 (SGD 입력 형식)
# pt 행렬을 (user_id, title, rating) 리스트 형태로 변환
R_df = filtered_data[['user_id', 'title', 'is_recommended']].copy()
R_df['rating'] = R_df['is_recommended'].astype(float) # 평점 컬럼 준비

# 사용자 및 아이템 ID를 내부 인덱스로 매핑
user_to_index = {u: i for i, u in enumerate(R_df['user_id'].unique())}
item_to_index = {i: j for j, i in enumerate(R_df['title'].unique())}
index_to_item = {v: k for k, v in item_to_index.items()}

R_df['u_idx'] = R_df['user_id'].map(user_to_index)
R_df['i_idx'] = R_df['title'].map(item_to_index)


# ==============================================================================
# 2. Matrix Factorization (MF) with SGD 클래스 정의 (이전 코드 재활용)
# ==============================================================================

class MatrixFactorizationSGD:
    def __init__(self, R, K, lr=0.01, reg=0.1, epochs=20):
        self.R = R
        self.K = K
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        
        self.num_users = R['u_idx'].nunique()
        self.num_items = R['i_idx'].nunique()
        
        # P: 사용자 잠재 요인, Q: 아이템 잠재 요인
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # 전역 평균 및 편향 초기화
        self.b_global = self.R['rating'].mean()
        self.b_user = np.zeros(self.num_users)
        self.b_item = np.zeros(self.num_items)

    def predict_rating(self, u, i):
        """특정 사용자와 아이템의 평점(추천 점수) 예측"""
        if u >= self.num_users or i >= self.num_items or u < 0 or i < 0:
            return self.b_global
            
        prediction = (
            self.b_global + 
            self.b_user[u] + 
            self.b_item[i] + 
            np.dot(self.P[u, :], self.Q[i, :])
        )
        # 평점 예측값을 0~1 사이로 제한 (이진 추천 점수이므로)
        return np.clip(prediction, 0.0, 1.0) 

    def train(self):
        """SGD를 사용하여 잠재 요인과 편향을 학습"""
        training_records = self.R.sample(frac=1).itertuples(index=False)
        for record in training_records:
            u, i, r = record.u_idx, record.i_idx, record.rating
            prediction = self.predict_rating(u, i)
            error = r - prediction
            
            # SGD 업데이트 규칙 적용 (잠재 요인 및 편향 업데이트)
            self.P[u, :] += self.lr * (error * self.Q[i, :] - self.reg * self.P[u, :])
            self.Q[i, :] += self.lr * (error * self.P[u, :] - self.reg * self.Q[i, :])
            self.b_user[u] += self.lr * (error - self.reg * self.b_user[u])
            self.b_item[i] += self.lr * (error - self.reg * self.b_item[i])
            
    def get_top_n_recommendations(self, user_id, n=5):
        """특정 사용자에게 Top-N 아이템을 추천"""
        if user_id not in user_to_index:
            return f"Error: User ID {user_id} not known."
            
        u_idx = user_to_index[user_id]
        
        # 사용자가 이미 플레이한 게임 인덱스 확인
        rated_games_indices = R_df[R_df['u_idx'] == u_idx]['i_idx'].unique()
        
        # 모든 게임에 대한 예측 평점 계산
        all_item_indices = np.arange(self.num_items)
        predictions = np.array([self.predict_rating(u_idx, i) for i in all_item_indices])
        
        # 이미 평가한 게임의 예측 점수를 최소값으로 설정하여 제외
        predictions[rated_games_indices] = -np.inf
        
        # 예측 점수가 높은 순으로 상위 N개 아이템 인덱스 추출
        top_n_indices = np.argsort(predictions)[::-1][:n]
        
        # 원본 Game Title로 변환하여 반환
        recommended_game_titles = [index_to_item[i] for i in top_n_indices]
        
        return recommended_game_titles


# ==============================================================================
# 3. 모델 학습 및 Top-N 추천 실행
# ==============================================================================

# 모델 학습 설정
LATENT_FACTORS = 50 
MF_model = MatrixFactorizationSGD(R_df, K=LATENT_FACTORS, lr=0.01, reg=0.05, epochs=10) # 필터링했으므로 epoch 수 감소 가능

print("\n--- 모델 학습 시작 (SGD Matrix Factorization) ---")
for epoch in range(MF_model.epochs):
    MF_model.train()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{MF_model.epochs} 완료")
print("--- 학습 완료 ---")

# 특정 사용자에게 Top-N 추천 실행 (필터링된 사용자 중 한 명을 선택)
# 필터링된 사용자의 ID를 무작위로 하나 선택하여 시연합니다.
target_user_id = R_df['user_id'].iloc[0] 
N_RECOMMEND = 5

recommendations = MF_model.get_top_n_recommendations(target_user_id, N_RECOMMEND)

# 사용자가 이미 플레이한 게임 목록 (참고용)
played_games = R_df[R_df['user_id'] == target_user_id]['title'].unique().tolist()

print("\n=======================================================")
print(f"🔥 User {target_user_id}의 Top-{N_RECOMMEND} 게임 추천 목록:")
print(f"  > 이미 플레이한 게임 ({len(played_games)}개): {played_games[:3]}...")
print("-------------------------------------------------------")
print(f"  > 추천 게임 (Top {N_RECOMMEND}):")
for i, title in enumerate(recommendations):
    print(f"    {i+1}. {title}")
print("=======================================================")