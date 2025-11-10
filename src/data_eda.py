import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (선택사항)
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
df = pd.read_csv('../data/merged_data.csv', na_values=['', ' ', 'NA', 'N/A', 'null', 'None'])

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
print("EDA 완료! 시각화를 생성합니다...")
print("=" * 50)

# 시각화 저장 디렉토리 생성
import os
output_dir = '../results/eda_visualizations'
os.makedirs(output_dir, exist_ok=True)

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ========================================
# 1. 결측치 히트맵
# ========================================
plt.figure(figsize=(12, 8))
missing_data = df.isnull()
sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Columns')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_missing_values_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 1. 결측치 히트맵 저장 완료")

# ========================================
# 2. Rating 분포
# ========================================
if 'rating' in df.columns and df['rating'].notna().any():
    plt.figure(figsize=(10, 6))
    rating_counts = df['rating'].value_counts().sort_index()
    plt.bar(range(len(rating_counts)), rating_counts.values, color='steelblue', alpha=0.8)
    plt.xticks(range(len(rating_counts)), rating_counts.index, rotation=45, ha='right')
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.title('Distribution of Game Ratings', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 2. Rating 분포 저장 완료")

# ========================================
# 3. Positive Ratio 분포
# ========================================
if 'positive_ratio' in df.columns and df['positive_ratio'].notna().any():
    plt.figure(figsize=(10, 6))
    positive_ratio_clean = df['positive_ratio'].dropna()
    plt.hist(positive_ratio_clean, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Positive Ratio (%)', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.title('Distribution of Positive Review Ratio', fontsize=16, fontweight='bold')
    plt.axvline(positive_ratio_clean.median(), color='red', linestyle='--',
                label=f'Median: {positive_ratio_clean.median():.1f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_positive_ratio_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 3. Positive Ratio 분포 저장 완료")

# ========================================
# 4. User Reviews 분포 (로그 스케일)
# ========================================
if 'user_reviews' in df.columns and df['user_reviews'].notna().any():
    plt.figure(figsize=(10, 6))
    user_reviews_clean = df['user_reviews'].dropna()
    user_reviews_clean = user_reviews_clean[user_reviews_clean > 0]
    plt.hist(np.log10(user_reviews_clean + 1), bins=50, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Log10(User Reviews + 1)', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.title('Distribution of User Reviews (Log Scale)', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_user_reviews_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 4. User Reviews 분포 저장 완료")

# ========================================
# 5. 가격 분포
# ========================================
if 'price_final' in df.columns and df['price_final'].notna().any():
    plt.figure(figsize=(10, 6))
    price_clean = df['price_final'].dropna()
    price_clean = price_clean[price_clean > 0]  # 무료 게임 제외
    plt.hist(price_clean, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.title('Distribution of Game Prices (Paid Games Only)', fontsize=16, fontweight='bold')
    plt.axvline(price_clean.median(), color='red', linestyle='--',
                label=f'Median: ${price_clean.median():.2f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 5. 가격 분포 저장 완료")

# ========================================
# 6. 플랫폼별 게임 수
# ========================================
platform_cols = ['win', 'mac', 'linux', 'steam_deck']
available_platforms = [col for col in platform_cols if col in df.columns]
if available_platforms:
    plt.figure(figsize=(10, 6))
    platform_counts = []
    platform_names = []
    for col in available_platforms:
        if df[col].notna().any():
            count = df[col].sum() if df[col].dtype == 'bool' else (df[col] == True).sum()
            platform_counts.append(count)
            platform_names.append(col.capitalize())

    plt.bar(platform_names, platform_counts, color=['#0078D4', '#999999', '#FCC624', '#1A1A1A'], alpha=0.8)
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.title('Games by Platform', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(platform_counts):
        plt.text(i, v + max(platform_counts)*0.01, str(int(v)), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_platform_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 6. 플랫폼별 게임 수 저장 완료")

# ========================================
# 7. 출시 연도별 게임 트렌드
# ========================================
if 'date_release' in df.columns and df['date_release'].notna().any():
    df_temp = df.copy()
    df_temp['date_release'] = pd.to_datetime(df_temp['date_release'], errors='coerce')
    df_temp['year'] = df_temp['date_release'].dt.year
    year_counts = df_temp['year'].value_counts().sort_index()
    year_counts = year_counts[(year_counts.index >= 2000) & (year_counts.index <= 2024)]

    plt.figure(figsize=(12, 6))
    plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=6, color='darkblue')
    plt.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='lightblue')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Games Released', fontsize=12)
    plt.title('Game Releases by Year (2000-2024)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_releases_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 7. 출시 연도별 트렌드 저장 완료")

# ========================================
# 8. Top 20 인기 게임 (리뷰 수 기준)
# ========================================
if 'title' in df.columns and 'user_reviews' in df.columns:
    df_games = df[['title', 'user_reviews']].drop_duplicates(subset='title').dropna()
    top_games = df_games.nlargest(20, 'user_reviews')

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_games)), top_games['user_reviews'].values, color='coral', alpha=0.8)
    plt.yticks(range(len(top_games)), top_games['title'].values, fontsize=10)
    plt.xlabel('Number of User Reviews', fontsize=12)
    plt.title('Top 20 Most Reviewed Games', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_top_20_games.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 8. Top 20 인기 게임 저장 완료")

# ========================================
# 9. Positive Ratio vs User Reviews (Scatter)
# ========================================
if 'positive_ratio' in df.columns and 'user_reviews' in df.columns:
    df_scatter = df[['positive_ratio', 'user_reviews']].dropna()
    df_scatter = df_scatter[df_scatter['user_reviews'] > 0]

    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(df_scatter['user_reviews'] + 1), df_scatter['positive_ratio'],
                alpha=0.3, s=10, color='darkgreen')
    plt.xlabel('Log10(User Reviews + 1)', fontsize=12)
    plt.ylabel('Positive Ratio (%)', fontsize=12)
    plt.title('Positive Ratio vs Number of Reviews', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_ratio_vs_reviews.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 9. Positive Ratio vs Reviews 저장 완료")

# ========================================
# 10. is_positive 분포 (리뷰 긍정/부정)
# ========================================
if 'is_positive' in df.columns and df['is_positive'].notna().any():
    plt.figure(figsize=(8, 6))
    sentiment_counts = df['is_positive'].value_counts()
    colors = ['#FF6B6B' if 'Negative' in str(x) else '#4ECDC4' for x in sentiment_counts.index]
    plt.bar(range(len(sentiment_counts)), sentiment_counts.values, color=colors, alpha=0.8)
    plt.xticks(range(len(sentiment_counts)), sentiment_counts.index, rotation=0)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.title('Distribution of Review Sentiments', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(sentiment_counts.values):
        plt.text(i, v + max(sentiment_counts.values)*0.01,
                f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 10. 리뷰 감정 분포 저장 완료")

print("\n" + "=" * 50)
print(f"모든 시각화 완료! 저장 위치: {output_dir}/")
print("=" * 50)
