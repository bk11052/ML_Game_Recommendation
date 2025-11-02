import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (선택사항)
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
df = pd.read_csv('merged_data.csv', na_values=['', ' ', 'NA', 'N/A', 'null', 'None'])

print("=" * 50)
print("1. 데이터 기본 정보")
print("=" * 50)
print(f"데이터 shape: {df.shape}")
print(f"행(rows): {len(df)}, 열(columns): {len(df.columns)}")

print("\n" + "=" * 50)
print("2. 데이터 미리보기")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("3. 컬럼 정보")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("4. 결측치 현황")
print("=" * 50)
missing = df.isnull().sum()
missing_percent = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '결측치 개수': missing,
    '결측치 비율(%)': missing_percent
})
print(missing_df[missing_df['결측치 개수'] > 0].sort_values('결측치 개수', ascending=False))

print("\n" + "=" * 50)
print("5. 수치형 변수 통계")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("6. 범주형 변수 분포")
print("=" * 50)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols[:5]:  # 처음 5개 범주형 변수만
    print(f"\n[{col}]")
    print(df[col].value_counts().head(10))

print("\n" + "=" * 50)
print("7. 데이터 타입")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("8. 중복 데이터 확인")
print("=" * 50)
print(f"중복 행 개수: {df.duplicated().sum()}")

print("\n" + "=" * 50)
print("EDA 완료!")
print("=" * 50)

# TODO: 시각화 추가
# - 결측치 히트맵
# - 수치형 변수 분포
# - 범주형 변수 분포
# - 변수 간 상관관계
