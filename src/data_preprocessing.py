import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 병합된 데이터 읽기
merged_df = pd.read_csv('../data/merged_data.csv')

print(f"원본 데이터 행 수: {len(merged_df)}")
print(f"원본 데이터 열 수: {len(merged_df.columns)}")
print(f"\n원본 데이터 정보:")
merged_df.info()

print("\n" + "="*30)
print("데이터 전처리를 시작합니다...")
print("="*30)

# 원본 데이터를 복사하여 전처리 진행
preprocessed_df = merged_df.copy()

## 1. 결측치 및 중복 데이터 처리

# 1-1. 중복 제거: 'id' (리뷰 ID)를 기준으로 중복된 행을 제거합니다.
print(f"\n[1-1] 중복 제거 전 행 수: {len(preprocessed_df)}")
preprocessed_df.drop_duplicates(subset=['id'], inplace=True)
print(f"중복 제거 후 행 수: {len(preprocessed_df)}")

# 1-2. 필수 컬럼 결측치 제거: 텍스트 분석에 필수적인 'content'가 없는 행을 제거합니다.
print(f"\n[1-2] 'content' 결측치 제거 전 행 수: {len(preprocessed_df)}")
preprocessed_df.dropna(subset=['content'], inplace=True)
print(f"'content' 결측치 제거 후 행 수: {len(preprocessed_df)}")

# 1-3. 결측치 확인 (참고)
# 스니펫에서 'title' 이하 게임 메타데이터에 결측치가 많은 것을 확인했습니다.
# 만약 게임 메타데이터 분석이 필수라면, 해당 결측치를 가진 행을 제거해야 합니다.
# 예: preprocessed_df.dropna(subset=['title', 'date_release'], inplace=True)
# 여기서는 텍스트 리뷰 분석을 가정하고, 'content'가 있는 행은 유지합니다.
print("\n[1-3] 컬럼별 결측치 개수 확인:")
print(preprocessed_df.isnull().sum())

# 1-4. 숫자형 컬럼 결측치 채우기 (예: 0으로 채우기)
# 'price_final' 등 숫자형 컬럼의 결측치는 0으로 채울 수 있습니다.
num_cols_to_fill = ['positive_ratio', 'user_reviews', 'price_final', 'price_original', 'discount']
for col in num_cols_to_fill:
    if col in preprocessed_df.columns:
        # 'user_reviews'와 같이 0이 적절한 값으로 채움
        preprocessed_df[col] = preprocessed_df[col].fillna(0)

print(f"\n[1-4] 숫자형 컬럼 결측치 0으로 채우기 완료.")

## 2. 데이터 타입 변환

print("\n[2] 데이터 타입 변환 시작...")

# 2-1. 날짜 타입 변환
if 'date_release' in preprocessed_df.columns:
    preprocessed_df['date_release'] = pd.to_datetime(preprocessed_df['date_release'], errors='coerce')
    print("- 'date_release' -> datetime 타입으로 변환 (오류는 NaT로 처리)")

# 2-2. 불리언(boolean) 타입 변환
bool_cols = ['win', 'mac', 'linux', 'steam_deck']
for col in bool_cols:
    if col in preprocessed_df.columns:
        # 결측치는 False로 간주
        preprocessed_df[col] = preprocessed_df[col].fillna('False') 
        preprocessed_df[col] = preprocessed_df[col].map({'True': True, 'False': False})
        print(f"- '{col}' -> bool 타입으로 변환 (NaN은 False로 처리)")

# 2-3. 숫자형(정수) 타입 변환
if 'user_reviews' in preprocessed_df.columns:
    # 결측치를 0으로 채웠으므로 int로 변환 가능
    preprocessed_df['user_reviews'] = preprocessed_df['user_reviews'].astype(int)
    print(f"- 'user_reviews' -> int 타입으로 변환")

## 3. 인코딩 (Encoding)

print("\n[3] 인코딩 시작...")

# 3-1. 'is_positive' (범주형 -> 숫자형)
# 'Positive'는 1, 'Negative'는 0으로 매핑
if 'is_positive' in preprocessed_df.columns:
    print(f"- 'is_positive' 고유 값: {preprocessed_df['is_positive'].unique()}")
    preprocessed_df['is_positive_encoded'] = preprocessed_df['is_positive'].map({'Positive': 1, 'Negative': 0})
    # 만약 'Positive', 'Negative' 외의 값이 있다면 NaN이 될 수 있으므로 확인
    if preprocessed_df['is_positive_encoded'].isnull().any():
        print("  [경고] 'is_positive'에 Positive/Negative 외의 값이 존재합니다.")
    print("- 'is_positive_encoded' (Positive: 1, Negative: 0) 컬럼 생성")

# 3-2. 'rating' (명목형) - 원-핫 인코딩 

preprocessed_df = pd.get_dummies(preprocessed_df, columns=['rating'], dummy_na=True, prefix='rating')


## 4. 이상치 제거 (Outlier Removal)

print("\n[4] 이상치 제거 (예시)...")

# 4-1. 'positive_ratio' (0 ~ 100 사이 값)
if 'positive_ratio' in preprocessed_df.columns:
    original_len = len(preprocessed_df)
    preprocessed_df = preprocessed_df[
        (preprocessed_df['positive_ratio'] >= 0) & (preprocessed_df['positive_ratio'] <= 100)
    ]
    print(f"- 'positive_ratio'가 0~100 범위를 벗어난 {original_len - len(preprocessed_df)}개 행 제거")

# 4-2. 'price_final' (0 이상 값)

if 'price_final' in preprocessed_df.columns:
    original_len = len(preprocessed_df)
    preprocessed_df = preprocessed_df[preprocessed_df['price_final'] >= 0]
    print(f"- 'price_final'이 0 미만인 {original_len - len(preprocessed_df)}개 행 제거")


## 5. 정규화/스케일링 (Normalization/Scaling)

scaler = MinMaxScaler()
cols_to_scale = ['user_reviews', 'price_final', 'positive_ratio']
 
# 스케일링할 컬럼이 DataFrame에 있는지 확인
existing_cols_to_scale = [col for col in cols_to_scale if col in preprocessed_df.columns]
 
if existing_cols_to_scale:
    print(f"- {existing_cols_to_scale} 컬럼에 대해 MinMaxScaler 적용 (주석 처리됨)")
    preprocessed_df[existing_cols_to_scale] = scaler.fit_transform(preprocessed_df[existing_cols_to_scale])


print("\n" + "="*30)
print("데이터 전처리 완료!")
print("="*30)

# 전처리된 데이터 정보 확인
print(f"\n전처리된 데이터 행 수: {len(preprocessed_df)}")
print(f"전처리된 데이터 열 수: {len(preprocessed_df.columns)}")
print(f"\n전처리된 데이터 정보:")
preprocessed_df.info()

# 전처리된 데이터 저장
preprocessed_df.to_csv('../data/preprocessed_data.csv', index=False)
print("\n'../data/preprocessed_data.csv' 파일로 저장 완료!")
