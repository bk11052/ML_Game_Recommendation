# ==============================================================================
# 모델명: Factorization Machines (FM) 통합 피처셋 구성
# ==============================================================================
# 사용 목적 (Why):
#   - CF(유저/아이템 ID)와 CBF(콘텐츠 특징) 정보를 따로 계산하지 않고, 모든 피처를 하나의 행렬로 통합하여 FM 모델에 입력하기 위함입니다.
#   - FM은 이 모든 피처들의 **쌍별 상호작용**을 학습하여, 50개 아이템 환경에서 CF와 CBF의 장점을 동시에 극대화합니다. (예: '이 유저 ID'와 '저 콘텐츠 키워드' 간의 상호작용)

# 사용 방법 (How):
#   - **CF 피처:** 유저 ID와 아이템 ID를 LabelBinarizer(sparse_output=True)를 사용해 메모리 효율적인 희소 원-핫 인코딩으로 변환합니다. (메모리 오류 해결)
#   - **CBF 피처:** 게임 'title'을 TF-IDF 벡터로 변환합니다.
#   - **통합:** CF 피처 희소 행렬과 CBF 피처 희소 행렬을 `hstack()` 함수로 수평 결합하여 최종 FM 입력 데이터를 완성합니다.

# 장단점:
#   - 장점: 모든 피처를 융합하므로, 이론상 세 가지 기법 중 가장 정교하고 강력한 예측 성능을 제공합니다. 희소성과 콘텐츠 정보를 동시에 가장 우아하게 처리합니다.
#   - 단점: 모델 자체의 구현 및 학습이 복잡하고, 전용 라이브러리(pyFM 등)가 필요합니다. 또한, 통합된 피처의 차원 수가 매우 높아 학습 시간이 오래 걸립니다.
# ==============================================================================


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import re
import random

# ==============================================================================
# 1. 데이터 로드 및 전처리 (이전과 동일)
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


# ==============================================================================
# 2. CF 피처 생성 (UserID, ItemID) - 메모리 오류 수정됨
# ==============================================================================

print("2. CF 피처(UserID, ItemID) 생성...")

# 메모리 오류 수정: sparse_output=True 설정
user_encoder = LabelBinarizer(sparse_output=True)
item_encoder = LabelBinarizer(sparse_output=True)

# CF 모델처럼 각 ID에 인덱스 부여
ratings_df_final['user_idx'] = pd.Categorical(ratings_df_final['author_id']).codes
ratings_df_final['item_idx'] = pd.Categorical(ratings_df_final['app_id']).codes

# 유저/아이템 피처를 희소 행렬로 변환 (이제 메모리 효율적임)
user_features = user_encoder.fit_transform(ratings_df_final['user_idx'])
item_features = item_encoder.fit_transform(ratings_df_final['item_idx'])

# CF 피처 결합
cf_features = hstack([user_features, item_features])
print(f"   -> CF 피처 희소 행렬 크기: {cf_features.shape}")


# ==============================================================================
# 3. CBF 피처 생성 (Title만 사용) - KeyError 수정됨
# ==============================================================================

print("3. CBF 피처(콘텐츠) 생성...")

# 3.1. 모든 유저 상호작용에 대해 게임 정보 결합
# (이전 KeyError로 인해 title, developer, publisher 중 title만 사용할 수 있음을 가정)
df_merged = pd.merge(ratings_df_final, df_games[['app_id', 'title']], on='app_id', how='left')

# 3.2. 콘텐츠 텍스트 통합 및 정리
def clean_text(text):
    if pd.isna(text): return ''
    text = str(text).lower().replace(' ', '')
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

# KeyError 수정: title만 사용
df_merged['content_text'] = df_merged['title'].fillna('').apply(clean_text)

# 3.3. TF-IDF 벡터화 (CBF 피처)
tfidf = TfidfVectorizer(token_pattern=r'\b\w{2,}\b')
cbf_features = tfidf.fit_transform(df_merged['content_text'])
print(f"   -> CBF 피처 희소 행렬 크기: {cbf_features.shape}")


# ==============================================================================
# 4. FM 통합 피처셋 구성
# ==============================================================================

# 최종 FM 입력 피처: CF 피처와 CBF 피처를 희소 행렬로 수평 결합
fm_input_features = hstack([cf_features, cbf_features])

# 최종 레이블(평점)
labels = df_merged['rating'].values

print("\n" + "="*80)
print("🎉 **Factorization Machines 통합 피처셋 구성 완료**")
print("="*80)
print(f"총 상호작용(리뷰) 수: {len(labels)}")
print(f"통합 피처 매트릭스 크기 (상호작용 수 x 총 피처 차원): {fm_input_features.shape}")

# 피처 구성 요약
n_users_final = user_features.shape[1]
n_items_final = item_features.shape[1]
n_cbf_final = cbf_features.shape[1]

print(f"   - CF (유저) 피처 수: {n_users_final}")
print(f"   - CF (아이템) 피처 수: {n_items_final}")
print(f"   - CBF (콘텐츠) 피처 수: {n_cbf_final}")
print("-" * 40)

print("**분석:** 이 코드는 메모리 효율적인 희소 행렬 형태로 FM 모델 입력 데이터를 구성했습니다.")
