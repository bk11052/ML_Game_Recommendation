import pandas as pd
import numpy as np

# 병합된 데이터 읽기
merged_df = pd.read_csv('merged_data.csv')

print(f"원본 데이터 행 수: {len(merged_df)}")
print(f"원본 데이터 열 수: {len(merged_df.columns)}")
print(f"\n데이터 정보:")
print(merged_df.info())

# TODO: 전처리 작업 추가
# 1. 결측치 처리
# 2. 데이터 타입 변환
# 3. 이상치 제거
# 4. 정규화/스케일링
# 5. 인코딩

# 전처리된 데이터 저장
# preprocessed_df.to_csv('preprocessed_data.csv', index=False)

print("\n전처리 준비 완료!")
