import pandas as pd

# CSV 파일 읽기
games_df = pd.read_csv('games.csv')
output_df = pd.read_csv('output.csv')

# app_id를 기준으로 두 데이터프레임 병합
# how='right': games.csv의 모든 게임 포함 (리뷰가 있는 게임은 상세 리뷰 정보도 포함)
merged_df = pd.merge(output_df, games_df, on='app_id', how='right')

# 병합된 데이터 저장
merged_df.to_csv('merged_data.csv', index=False)

print(f"병합 완료!")
print(f"games.csv 행 수: {len(games_df)}")
print(f"output.csv 행 수: {len(output_df)}")
print(f"merged_data.csv 행 수: {len(merged_df)}")
print(f"\n병합된 데이터 미리보기:")
print(merged_df.head())
