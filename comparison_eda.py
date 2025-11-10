"""
DATA INTEGRATION 전후 비교 분석 스크립트

이 스크립트는 데이터 병합 전(games.csv, output.csv)과
병합 후(merged_data.csv)를 비교 분석합니다.

사용 방법:
1. data/ 폴더에 다음 파일들이 필요합니다:
   - games.csv (data/integration 브랜치에서 가져오기)
   - output.csv (data/integration 브랜치에서 가져오기)
   - merged_data.csv (이미 존재)

2. 파일 준비:
   git checkout data/integration -- games.csv output.csv
   mv games.csv output.csv data/

3. 실행:
   python comparison_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 80)
print("DATA INTEGRATION 전후 비교 분석")
print("=" * 80)

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# 결과 저장 디렉토리 생성
output_dir = 'comparison_results'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n[1] 데이터 로드 중...")

games_df = pd.read_csv('data/games.csv')
output_df = pd.read_csv('data/output.csv')
merged_df = pd.read_csv('data/merged_data.csv')

print("✓ 모든 데이터 로드 완료")

# ============================================================================
# 2. 기본 정보 비교
# ============================================================================
print("\n" + "=" * 80)
print("[2] 기본 정보 비교")
print("=" * 80)

comparison_data = {
    '파일명': ['games.csv', 'output.csv', 'merged_data.csv'],
    '행(rows)': [len(games_df), len(output_df), len(merged_df)],
    '열(columns)': [len(games_df.columns), len(output_df.columns), len(merged_df.columns)],
    '메모리 사용량': [
        f"{games_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        f"{output_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        f"{merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    ]
}

comparison_summary = pd.DataFrame(comparison_data)
print("\n📊 데이터셋 크기 비교:")
print(comparison_summary.to_string(index=False))

# ============================================================================
# 3. games.csv 분석
# ============================================================================
print("\n" + "=" * 80)
print("[3] games.csv 분석 (게임 메타데이터)")
print("=" * 80)

print(f"\n총 게임 수: {len(games_df):,}개")
print(f"컬럼 수: {len(games_df.columns)}개")
print(f"\n컬럼 목록:")
print(games_df.columns.tolist())

print(f"\n기본 통계:")
print(games_df.info())

print(f"\n결측치:")
print(games_df.isnull().sum())

if 'rating' in games_df.columns:
    print(f"\n평점(rating) 분포:")
    print(games_df['rating'].value_counts())

if 'user_reviews' in games_df.columns:
    print(f"\n리뷰 수 통계:")
    print(games_df['user_reviews'].describe())

    # Top 10 가장 많은 리뷰를 받은 게임
    print(f"\n📌 리뷰 수 Top 10 게임:")
    top_reviewed = games_df.nlargest(10, 'user_reviews')[['title', 'user_reviews', 'rating', 'positive_ratio']]
    for idx, row in top_reviewed.iterrows():
        print(f"   {row['title'][:50]:50s} | 리뷰: {row['user_reviews']:>8,}개 | 평점: {row['rating']:20s} | 긍정률: {row['positive_ratio']:>3}%")

# 가격 분석
if 'price_final' in games_df.columns:
    print(f"\n💰 가격 통계:")
    print(games_df['price_final'].describe())
    free_games = (games_df['price_final'] == 0).sum()
    paid_games = (games_df['price_final'] > 0).sum()
    print(f"\n   - 무료 게임: {free_games:,}개 ({free_games/len(games_df)*100:.1f}%)")
    print(f"   - 유료 게임: {paid_games:,}개 ({paid_games/len(games_df)*100:.1f}%)")
    if paid_games > 0:
        avg_price = games_df[games_df['price_final'] > 0]['price_final'].mean()
        median_price = games_df[games_df['price_final'] > 0]['price_final'].median()
        print(f"   - 유료 게임 평균 가격: ${avg_price:.2f}")
        print(f"   - 유료 게임 중간 가격: ${median_price:.2f}")

# 플랫폼 분석
platform_cols = ['win', 'mac', 'linux', 'steam_deck']
available_platforms = [col for col in platform_cols if col in games_df.columns]
if available_platforms:
    print(f"\n🖥️  플랫폼 지원:")
    for col in available_platforms:
        count = games_df[col].sum()
        print(f"   - {col.upper():11s}: {count:>6,}개 ({count/len(games_df)*100:>5.1f}%)")

# 출시 연도 분석
if 'date_release' in games_df.columns:
    games_temp = games_df.copy()
    games_temp['date_release'] = pd.to_datetime(games_temp['date_release'], errors='coerce')
    games_temp['year'] = games_temp['date_release'].dt.year
    year_valid = games_temp['year'].notna().sum()
    print(f"\n📅 출시 연도 정보:")
    print(f"   - 유효한 날짜: {year_valid:,}개")
    if year_valid > 0:
        print(f"   - 최초 출시: {int(games_temp['year'].min())}년")
        print(f"   - 최근 출시: {int(games_temp['year'].max())}년")
        most_common_year = games_temp['year'].mode()[0] if len(games_temp['year'].mode()) > 0 else None
        if most_common_year:
            year_count = (games_temp['year'] == most_common_year).sum()
            print(f"   - 가장 많은 출시 연도: {int(most_common_year)}년 ({year_count:,}개)")

# ============================================================================
# 4. output.csv 분석
# ============================================================================
print("\n" + "=" * 80)
print("[4] output.csv 분석 (리뷰 데이터)")
print("=" * 80)

print(f"\n총 리뷰 수: {len(output_df):,}개")
print(f"컬럼 수: {len(output_df.columns)}개")
print(f"\n컬럼 목록:")
print(output_df.columns.tolist())

print(f"\n기본 통계:")
print(output_df.info())

print(f"\n결측치:")
print(output_df.isnull().sum())

if 'is_positive' in output_df.columns:
    print(f"\n긍정/부정 리뷰 분포:")
    print(output_df['is_positive'].value_counts())

if 'app_id' in output_df.columns:
    unique_games_with_reviews = output_df['app_id'].nunique()
    print(f"\n리뷰가 있는 고유 게임 수: {unique_games_with_reviews:,}개")
    print(f"게임당 평균 리뷰 수: {len(output_df) / unique_games_with_reviews:.2f}개")

    # 게임별 리뷰 수 분석
    reviews_per_game = output_df['app_id'].value_counts()
    print(f"\n📊 게임별 리뷰 수 분포:")
    print(f"   - 최소: {reviews_per_game.min():,}개")
    print(f"   - 최대: {reviews_per_game.max():,}개")
    print(f"   - 평균: {reviews_per_game.mean():.1f}개")
    print(f"   - 중간값: {reviews_per_game.median():.1f}개")

    # 가장 많은 리뷰를 받은 게임
    print(f"\n📌 리뷰 수 Top 10 게임 (app_id):")
    top_games = reviews_per_game.head(10)
    for app_id, count in top_games.items():
        print(f"   app_id {app_id}: {count:>6,}개 리뷰")

# 리뷰 길이 분석
if 'content' in output_df.columns:
    content_lengths = output_df['content'].dropna().str.len()
    print(f"\n📝 리뷰 텍스트 길이:")
    print(f"   - 평균 길이: {content_lengths.mean():.1f}자")
    print(f"   - 중간 길이: {content_lengths.median():.1f}자")
    print(f"   - 최소 길이: {content_lengths.min()}자")
    print(f"   - 최대 길이: {content_lengths.max():,}자")

    # 길이별 분포
    very_short = (content_lengths < 10).sum()
    short = ((content_lengths >= 10) & (content_lengths < 50)).sum()
    medium = ((content_lengths >= 50) & (content_lengths < 200)).sum()
    long_text = (content_lengths >= 200).sum()
    total = len(content_lengths)

    print(f"\n   리뷰 길이 분포:")
    print(f"   - 매우 짧음 (<10자):    {very_short:>6,}개 ({very_short/total*100:>5.1f}%)")
    print(f"   - 짧음 (10-50자):        {short:>6,}개 ({short/total*100:>5.1f}%)")
    print(f"   - 보통 (50-200자):       {medium:>6,}개 ({medium/total*100:>5.1f}%)")
    print(f"   - 긴 리뷰 (200자+):      {long_text:>6,}개 ({long_text/total*100:>5.1f}%)")

# ============================================================================
# 5. merged_data.csv 분석
# ============================================================================
print("\n" + "=" * 80)
print("[5] merged_data.csv 분석 (병합 데이터)")
print("=" * 80)

print(f"\n총 행 수: {len(merged_df):,}개")
print(f"컬럼 수: {len(merged_df.columns)}개")
print(f"\n컬럼 목록:")
print(merged_df.columns.tolist())

print(f"\n기본 통계:")
print(merged_df.info())

print(f"\n결측치:")
missing_merged = merged_df.isnull().sum()
print(missing_merged[missing_merged > 0].sort_values(ascending=False))

# 결측치 비율
print(f"\n📊 결측치 비율:")
missing_with_ratio = missing_merged[missing_merged > 0].sort_values(ascending=False)
for col, count in missing_with_ratio.items():
    ratio = count / len(merged_df) * 100
    print(f"   - {col:15s}: {count:>7,}개 ({ratio:>5.2f}%)")

# 병합 결과 데이터 타입
print(f"\n📋 데이터 타입 분포:")
dtype_counts = merged_df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   - {str(dtype):10s}: {count:>2}개 컬럼")

# 리뷰가 있는 vs 없는 게임 분석
if 'id' in merged_df.columns:
    games_with_reviews = merged_df['id'].notna().sum()
    games_without_reviews = merged_df['id'].isna().sum()
    print(f"\n🎮 게임 데이터 분류:")
    print(f"   - 리뷰가 있는 행: {games_with_reviews:>7,}개 ({games_with_reviews/len(merged_df)*100:>5.1f}%)")
    print(f"   - 리뷰가 없는 행: {games_without_reviews:>7,}개 ({games_without_reviews/len(merged_df)*100:>5.1f}%)")

# 병합된 데이터의 게임 정보
if 'title' in merged_df.columns:
    unique_titles = merged_df['title'].nunique()
    print(f"\n   - 고유 게임 타이틀: {unique_titles:,}개")

    # 가장 많은 행을 차지하는 게임 (리뷰가 많은 게임)
    print(f"\n📌 가장 많은 리뷰가 있는 게임 Top 10:")
    top_reviewed_merged = merged_df['title'].value_counts().head(10)
    for title, count in top_reviewed_merged.items():
        # 해당 게임의 평점과 긍정률 가져오기
        game_info = merged_df[merged_df['title'] == title].iloc[0]
        rating = game_info['rating'] if 'rating' in merged_df.columns else 'N/A'
        pos_ratio = game_info['positive_ratio'] if 'positive_ratio' in merged_df.columns else 'N/A'
        print(f"   {title[:45]:45s} | {count:>6,}개 리뷰 | {str(rating):20s} | 긍정률: {pos_ratio}%")

# ============================================================================
# 6. 병합 전후 비교 분석
# ============================================================================
print("\n" + "=" * 80)
print("[6] 병합 전후 비교 분석")
print("=" * 80)

# 게임 수 비교
games_in_games_csv = len(games_df)
games_in_output_csv = output_df['app_id'].nunique() if 'app_id' in output_df.columns else 0
games_in_merged = merged_df['app_id'].nunique() if 'app_id' in merged_df.columns else 0

print(f"\n📊 게임 수 비교:")
print(f"  - games.csv 게임 수: {games_in_games_csv:,}개")
print(f"  - output.csv 리뷰가 있는 게임: {games_in_output_csv:,}개")
print(f"  - merged_data.csv 고유 게임: {games_in_merged:,}개")

# 리뷰 수 비교
reviews_in_output = len(output_df)
reviews_in_merged = merged_df['id'].notna().sum() if 'id' in merged_df.columns else 0

print(f"\n📊 리뷰 수 비교:")
print(f"  - output.csv 총 리뷰: {reviews_in_output:,}개")
print(f"  - merged_data.csv 리뷰 데이터: {reviews_in_merged:,}개")
print(f"  - 병합 후 리뷰가 없는 게임: {len(merged_df) - reviews_in_merged:,}개")

# 데이터 보존율
data_preservation = (reviews_in_merged / reviews_in_output * 100) if reviews_in_output > 0 else 0
print(f"\n📊 데이터 보존율:")
print(f"  - 리뷰 데이터 보존율: {data_preservation:.2f}%")
lost_reviews = reviews_in_output - reviews_in_merged
print(f"  - 손실된 리뷰: {lost_reviews:,}개")

# 컬럼 비교
print(f"\n📋 컬럼 변화:")
games_cols = set(games_df.columns)
output_cols = set(output_df.columns)
merged_cols = set(merged_df.columns)

print(f"  - games.csv 고유 컬럼: {len(games_cols - output_cols)}개")
print(f"    {sorted(list(games_cols - output_cols))}")
print(f"  - output.csv 고유 컬럼: {len(output_cols - games_cols)}개")
print(f"    {sorted(list(output_cols - games_cols))}")
print(f"  - merged_data.csv 전체 컬럼: {len(merged_cols)}개")

common_cols = games_cols & output_cols
print(f"  - 공통 컬럼 (병합 키): {len(common_cols)}개")
print(f"    {sorted(list(common_cols))}")

# 병합 품질 검증
print(f"\n🔍 병합 품질 검증:")
if 'app_id' in merged_df.columns:
    # 중복 app_id 확인
    duplicate_appids = merged_df[merged_df.duplicated(subset=['app_id'], keep=False)]
    if len(duplicate_appids) > 0:
        print(f"  ⚠️  중복된 app_id가 있는 행: {len(duplicate_appids):,}개")
        print(f"     → 이는 하나의 게임에 여러 리뷰가 있기 때문에 정상입니다.")
    else:
        print(f"  ✓ 중복 없음 (각 app_id는 unique)")

    # 병합 후 모든 게임이 유지되었는지 확인
    games_preserved = merged_df['app_id'].nunique() == games_in_games_csv
    if games_preserved:
        print(f"  ✓ 모든 게임이 보존됨 ({games_in_games_csv:,}개)")
    else:
        print(f"  ⚠️  일부 게임이 손실됨")

# 통계 비교
print(f"\n📈 주요 통계 비교:")
if 'positive_ratio' in games_df.columns and 'positive_ratio' in merged_df.columns:
    games_avg_ratio = games_df['positive_ratio'].mean()
    merged_avg_ratio = merged_df['positive_ratio'].mean()
    print(f"  긍정 비율 평균:")
    print(f"    - games.csv:       {games_avg_ratio:.2f}%")
    print(f"    - merged_data.csv: {merged_avg_ratio:.2f}%")
    print(f"    - 차이:            {abs(games_avg_ratio - merged_avg_ratio):.2f}%p")

if 'user_reviews' in games_df.columns and 'user_reviews' in merged_df.columns:
    games_avg_reviews = games_df['user_reviews'].mean()
    merged_avg_reviews = merged_df['user_reviews'].mean()
    print(f"\n  리뷰 수 평균:")
    print(f"    - games.csv:       {games_avg_reviews:,.1f}개")
    print(f"    - merged_data.csv: {merged_avg_reviews:,.1f}개")
    print(f"    - 차이:            {abs(games_avg_reviews - merged_avg_reviews):,.1f}개")

# ============================================================================
# 7. 시각화 생성
# ============================================================================
print("\n" + "=" * 80)
print("[7] 시각화 생성 중...")
print("=" * 80)

sns.set_style("whitegrid")

# 7-1. 데이터셋 크기 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 행 수 비교
datasets = ['games.csv', 'output.csv', 'merged_data.csv']
row_counts = [len(games_df), len(output_df), len(merged_df)]
colors = ['#3498db', '#e74c3c', '#2ecc71']

axes[0].bar(range(len(datasets)), row_counts, color=colors, alpha=0.8)
axes[0].set_xticks(range(len(datasets)))
axes[0].set_xticklabels(datasets, rotation=15, ha='right')
axes[0].set_ylabel('Number of Rows', fontsize=11)
axes[0].set_title('Dataset Size Comparison (Rows)', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(row_counts):
    axes[0].text(i, v + max(row_counts)*0.02, f'{v:,}', ha='center', fontweight='bold')

# 열 수 비교
col_counts = [len(games_df.columns), len(output_df.columns), len(merged_df.columns)]
axes[1].bar(range(len(datasets)), col_counts, color=colors, alpha=0.8)
axes[1].set_xticks(range(len(datasets)))
axes[1].set_xticklabels(datasets, rotation=15, ha='right')
axes[1].set_ylabel('Number of Columns', fontsize=11)
axes[1].set_title('Dataset Size Comparison (Columns)', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(col_counts):
    axes[1].text(i, v + max(col_counts)*0.02, f'{v}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/01_dataset_size_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 1. 데이터셋 크기 비교 저장 완료")

# 7-2. 게임 수 비교
plt.figure(figsize=(10, 6))
game_counts = {
    'games.csv\n(Total Games)': games_in_games_csv,
    'output.csv\n(Games with Reviews)': games_in_output_csv,
    'merged_data.csv\n(Unique Games)': games_in_merged
}

bars = plt.bar(range(len(game_counts)), list(game_counts.values()),
               color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, width=0.6)
plt.xticks(range(len(game_counts)), list(game_counts.keys()), fontsize=10)
plt.ylabel('Number of Games', fontsize=11)
plt.title('Game Count Comparison Across Datasets', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for i, (key, val) in enumerate(game_counts.items()):
    plt.text(i, val + max(game_counts.values())*0.02, f'{val:,}',
             ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_game_count_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 2. 게임 수 비교 저장 완료")

# 7-3. 리뷰 수 비교
plt.figure(figsize=(10, 6))
review_data = {
    'output.csv\n(Total Reviews)': reviews_in_output,
    'merged_data.csv\n(Reviews with Game Info)': reviews_in_merged,
    'merged_data.csv\n(Games without Reviews)': len(merged_df) - reviews_in_merged
}

colors_review = ['#e74c3c', '#2ecc71', '#95a5a6']
bars = plt.bar(range(len(review_data)), list(review_data.values()),
               color=colors_review, alpha=0.8, width=0.6)
plt.xticks(range(len(review_data)), list(review_data.keys()), fontsize=10)
plt.ylabel('Count', fontsize=11)
plt.title('Review Data Comparison', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for i, (key, val) in enumerate(review_data.items()):
    plt.text(i, val + max(review_data.values())*0.02, f'{val:,}',
             ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_review_count_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 3. 리뷰 수 비교 저장 완료")

# 7-4. 결측치 비교
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# games.csv 결측치
missing_games = games_df.isnull().sum()
if missing_games.sum() > 0:
    missing_games = missing_games[missing_games > 0].sort_values(ascending=False)
    axes[0].barh(range(len(missing_games)), missing_games.values, color='#3498db', alpha=0.8)
    axes[0].set_yticks(range(len(missing_games)))
    axes[0].set_yticklabels(missing_games.index, fontsize=9)
    axes[0].set_xlabel('Missing Count', fontsize=10)
    axes[0].set_title('games.csv - Missing Values', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
else:
    axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=12)
    axes[0].set_title('games.csv - Missing Values', fontsize=12, fontweight='bold')

# output.csv 결측치
missing_output = output_df.isnull().sum()
if missing_output.sum() > 0:
    missing_output = missing_output[missing_output > 0].sort_values(ascending=False)
    axes[1].barh(range(len(missing_output)), missing_output.values, color='#e74c3c', alpha=0.8)
    axes[1].set_yticks(range(len(missing_output)))
    axes[1].set_yticklabels(missing_output.index, fontsize=9)
    axes[1].set_xlabel('Missing Count', fontsize=10)
    axes[1].set_title('output.csv - Missing Values', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=12)
    axes[1].set_title('output.csv - Missing Values', fontsize=12, fontweight='bold')

# merged_data.csv 결측치
missing_merged_df = merged_df.isnull().sum()
missing_merged_df = missing_merged_df[missing_merged_df > 0].sort_values(ascending=False).head(10)
if len(missing_merged_df) > 0:
    axes[2].barh(range(len(missing_merged_df)), missing_merged_df.values, color='#2ecc71', alpha=0.8)
    axes[2].set_yticks(range(len(missing_merged_df)))
    axes[2].set_yticklabels(missing_merged_df.index, fontsize=9)
    axes[2].set_xlabel('Missing Count', fontsize=10)
    axes[2].set_title('merged_data.csv - Missing Values (Top 10)', fontsize=12, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
else:
    axes[2].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=12)
    axes[2].set_title('merged_data.csv - Missing Values', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/04_missing_values_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 4. 결측치 비교 저장 완료")

# 7-5. 긍정/부정 리뷰 비교 (output.csv vs merged_data.csv)
if 'is_positive' in output_df.columns and 'is_positive' in merged_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # output.csv
    sentiment_output = output_df['is_positive'].value_counts()
    colors_sentiment = ['#2ecc71' if 'Positive' in str(x) else '#e74c3c' for x in sentiment_output.index]
    axes[0].bar(range(len(sentiment_output)), sentiment_output.values,
                color=colors_sentiment, alpha=0.8)
    axes[0].set_xticks(range(len(sentiment_output)))
    axes[0].set_xticklabels(sentiment_output.index, fontsize=10)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('output.csv - Review Sentiment', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(sentiment_output.values):
        axes[0].text(i, v + max(sentiment_output.values)*0.02,
                    f'{v:,}\n({v/len(output_df)*100:.1f}%)',
                    ha='center', fontweight='bold')

    # merged_data.csv
    sentiment_merged = merged_df['is_positive'].value_counts()
    colors_sentiment_merged = ['#2ecc71' if 'Positive' in str(x) else '#e74c3c' for x in sentiment_merged.index]
    axes[1].bar(range(len(sentiment_merged)), sentiment_merged.values,
                color=colors_sentiment_merged, alpha=0.8)
    axes[1].set_xticks(range(len(sentiment_merged)))
    axes[1].set_xticklabels(sentiment_merged.index, fontsize=10)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('merged_data.csv - Review Sentiment', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(sentiment_merged.values):
        axes[1].text(i, v + max(sentiment_merged.values)*0.02,
                    f'{v:,}\n({v/sentiment_merged.sum()*100:.1f}%)',
                    ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_sentiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 5. 감정 분포 비교 저장 완료")

# 7-6. 평점 분포 비교 (games.csv vs merged_data.csv)
if 'rating' in games_df.columns and 'rating' in merged_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # games.csv
    rating_games = games_df['rating'].value_counts().sort_index()
    axes[0].barh(range(len(rating_games)), rating_games.values, color='#3498db', alpha=0.8)
    axes[0].set_yticks(range(len(rating_games)))
    axes[0].set_yticklabels(rating_games.index, fontsize=9)
    axes[0].set_xlabel('Count', fontsize=11)
    axes[0].set_title('games.csv - Rating Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()

    # merged_data.csv
    rating_merged = merged_df['rating'].value_counts().sort_index()
    axes[1].barh(range(len(rating_merged)), rating_merged.values, color='#2ecc71', alpha=0.8)
    axes[1].set_yticks(range(len(rating_merged)))
    axes[1].set_yticklabels(rating_merged.index, fontsize=9)
    axes[1].set_xlabel('Count', fontsize=11)
    axes[1].set_title('merged_data.csv - Rating Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_rating_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 6. 평점 분포 비교 저장 완료")

# ============================================================================
# 8. 요약 리포트 저장
# ============================================================================
print("\n" + "=" * 80)
print("[8] 요약 리포트 생성 중...")
print("=" * 80)

summary_report = f"""
================================================================================
DATA INTEGRATION 전후 비교 분석 요약 리포트
================================================================================

1. 데이터셋 크기
   - games.csv:       {len(games_df):,} rows × {len(games_df.columns)} columns
   - output.csv:      {len(output_df):,} rows × {len(output_df.columns)} columns
   - merged_data.csv: {len(merged_df):,} rows × {len(merged_df.columns)} columns

2. 게임 수
   - games.csv 총 게임:           {games_in_games_csv:,}개
   - output.csv 리뷰 있는 게임:   {games_in_output_csv:,}개
   - merged_data.csv 고유 게임:   {games_in_merged:,}개

3. 리뷰 수
   - output.csv 총 리뷰:          {reviews_in_output:,}개
   - merged_data.csv 리뷰 데이터: {reviews_in_merged:,}개
   - 병합 후 리뷰 없는 게임:      {len(merged_df) - reviews_in_merged:,}개

4. 데이터 보존율
   - 리뷰 데이터 보존율: {data_preservation:.2f}%

5. 병합 전략
   - 사용된 join 방식: RIGHT JOIN (games.csv 기준)
   - 모든 게임 포함: 리뷰가 없는 게임도 포함되어 결측치 존재

6. 주요 발견사항
   - games.csv는 모든 게임의 메타데이터를 포함
   - output.csv는 일부 게임에 대한 리뷰만 포함 ({games_in_output_csv:,}개 게임)
   - merged_data.csv는 right join으로 모든 게임 포함하되,
     리뷰가 없는 게임은 리뷰 관련 컬럼이 NaN

7. 시각화 파일
   - 01_dataset_size_comparison.png: 데이터셋 크기 비교
   - 02_game_count_comparison.png: 게임 수 비교
   - 03_review_count_comparison.png: 리뷰 수 비교
   - 04_missing_values_comparison.png: 결측치 비교
   - 05_sentiment_comparison.png: 감정 분포 비교
   - 06_rating_comparison.png: 평점 분포 비교

================================================================================
분석 완료 시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""

with open(f'{output_dir}/comparison_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)
print(f"\n✓ 요약 리포트 저장 완료: {output_dir}/comparison_summary.txt")

print("\n" + "=" * 80)
print(f"모든 분석 완료! 결과는 '{output_dir}/' 폴더에 저장되었습니다.")
print("=" * 80)
