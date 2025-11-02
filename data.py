import pandas as pd

# CSV 파일 읽기
games_df = pd.read_csv('games.csv')
output_df = pd.read_csv('output.csv')

print(games_df.head())