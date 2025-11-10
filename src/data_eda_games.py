"""
games.csv 전용 EDA 스크립트
게임 메타데이터 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
print("=" * 80)
print("GAMES.CSV - 게임 메타데이터 분석")
print("=" * 80)
print("\n데이터 로드 중...")

# 스크립트 위치에 따라 경로 자동 조정
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data', 'games.csv')

df = pd.read_csv(data_path)
print("✓ 데이터 로드 완료\n")

# ============================================================================
# 1. 기본 정보
# ============================================================================
print("=" * 80)
print("1. 기본 정보")
print("=" * 80)
print(f"총 게임 수: {len(df):,}개")
print(f"컬럼 수: {len(df.columns)}개")
print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n컬럼 목록:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# 2. 데이터 미리보기
# ============================================================================
print("\n" + "=" * 80)
print("2. 데이터 미리보기")
print("=" * 80)
print(df.head(10))

# ============================================================================
# 3. 컬럼 정보
# ============================================================================
print("\n" + "=" * 80)
print("3. 컬럼 상세 정보")
print("=" * 80)
print(df.info())

# ============================================================================
# 4. 결측치 분석
# ============================================================================
print("\n" + "=" * 80)
print("4. 결측치 분석")
print("=" * 80)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ 결측치 없음 - 완벽한 게임 메타데이터!")
else:
    print("결측치가 있는 컬럼:")
    for col, count in missing[missing > 0].items():
        print(f"  - {col}: {count:,}개 ({count/len(df)*100:.2f}%)")

# ============================================================================
# 5. 수치형 변수 통계
# ============================================================================
print("\n" + "=" * 80)
print("5. 수치형 변수 통계")
print("=" * 80)
print(df.describe())

# ============================================================================
# 6. 평점(Rating) 분석
# ============================================================================
print("\n" + "=" * 80)
print("6. 평점(Rating) 분석")
print("=" * 80)
if 'rating' in df.columns:
    rating_counts = df['rating'].value_counts().sort_values(ascending=False)
    print("\n평점 분포:")
    for rating, count in rating_counts.items():
        percentage = count / len(df) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {rating:25s}: {count:>6,}개 ({percentage:>5.1f}%) {bar}")

# ============================================================================
# 7. 긍정 비율(Positive Ratio) 분석
# ============================================================================
print("\n" + "=" * 80)
print("7. 긍정 비율(Positive Ratio) 분석")
print("=" * 80)
if 'positive_ratio' in df.columns:
    print(f"\n평균 긍정 비율: {df['positive_ratio'].mean():.2f}%")
    print(f"중간값: {df['positive_ratio'].median():.2f}%")
    print(f"최소: {df['positive_ratio'].min()}%")
    print(f"최대: {df['positive_ratio'].max()}%")

    # 구간별 분포
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['매우 부정적 (0-20%)', '부정적 (20-40%)', '중립 (40-60%)', '긍정적 (60-80%)', '매우 긍정적 (80-100%)']
    df['ratio_category'] = pd.cut(df['positive_ratio'], bins=bins, labels=labels, include_lowest=True)

    print("\n긍정 비율 구간별 분포:")
    for category, count in df['ratio_category'].value_counts().sort_index().items():
        percentage = count / len(df) * 100
        print(f"  {category:25s}: {count:>6,}개 ({percentage:>5.1f}%)")

# ============================================================================
# 8. 리뷰 수(User Reviews) 분석
# ============================================================================
print("\n" + "=" * 80)
print("8. 리뷰 수(User Reviews) 분석")
print("=" * 80)
if 'user_reviews' in df.columns:
    print(f"\n평균 리뷰 수: {df['user_reviews'].mean():,.1f}개")
    print(f"중간값: {df['user_reviews'].median():,.1f}개")
    print(f"최소: {df['user_reviews'].min():,}개")
    print(f"최대: {df['user_reviews'].max():,}개")

    # 리뷰 수 구간별 분포
    print("\n리뷰 수 구간별 게임 수:")
    ranges = [(10, 100), (100, 1000), (1000, 10000), (10000, 100000), (100000, float('inf'))]
    for low, high in ranges:
        count = ((df['user_reviews'] >= low) & (df['user_reviews'] < high)).sum()
        percentage = count / len(df) * 100
        print(f"  {low:>6,} ~ {high if high != float('inf') else '∞':>6} 개: {count:>6,}개 ({percentage:>5.1f}%)")

    # Top 20 리뷰 많은 게임
    print("\n📌 Top 20 리뷰가 많은 게임:")
    top_reviewed = df.nlargest(20, 'user_reviews')[['title', 'user_reviews', 'rating', 'positive_ratio']]
    for i, (idx, row) in enumerate(top_reviewed.iterrows(), 1):
        print(f"  {i:2d}. {row['title'][:45]:45s} | {row['user_reviews']:>10,}개 | {row['rating']:20s} | {row['positive_ratio']:>3}%")

# ============================================================================
# 9. 가격(Price) 분석
# ============================================================================
print("\n" + "=" * 80)
print("9. 가격(Price) 분석")
print("=" * 80)
if 'price_final' in df.columns:
    free_games = (df['price_final'] == 0).sum()
    paid_games = (df['price_final'] > 0).sum()

    print(f"\n무료 게임: {free_games:,}개 ({free_games/len(df)*100:.1f}%)")
    print(f"유료 게임: {paid_games:,}개 ({paid_games/len(df)*100:.1f}%)")

    if paid_games > 0:
        paid_df = df[df['price_final'] > 0]
        print(f"\n유료 게임 가격 통계:")
        print(f"  평균: ${paid_df['price_final'].mean():.2f}")
        print(f"  중간값: ${paid_df['price_final'].median():.2f}")
        print(f"  최소: ${paid_df['price_final'].min():.2f}")
        print(f"  최대: ${paid_df['price_final'].max():.2f}")

        # 가격 구간별 분포
        print("\n가격 구간별 게임 수:")
        price_ranges = [(0.01, 5), (5, 10), (10, 20), (20, 40), (40, float('inf'))]
        for low, high in price_ranges:
            count = ((paid_df['price_final'] >= low) & (paid_df['price_final'] < high)).sum()
            percentage = count / len(paid_df) * 100
            print(f"  ${low:>5.2f} ~ ${high if high != float('inf') else '∞':>5} : {count:>6,}개 ({percentage:>5.1f}%)")

    # 할인 분석
    if 'discount' in df.columns:
        discounted = (df['discount'] > 0).sum()
        print(f"\n할인 중인 게임: {discounted:,}개 ({discounted/len(df)*100:.1f}%)")
        if discounted > 0:
            print(f"평균 할인율: {df[df['discount'] > 0]['discount'].mean():.1f}%")
            print(f"최대 할인율: {df['discount'].max():.0f}%")

# ============================================================================
# 10. 플랫폼 지원 분석
# ============================================================================
print("\n" + "=" * 80)
print("10. 플랫폼 지원 분석")
print("=" * 80)
platform_cols = ['win', 'mac', 'linux', 'steam_deck']
available_platforms = [col for col in platform_cols if col in df.columns]

if available_platforms:
    print("\n플랫폼별 게임 수:")
    for col in available_platforms:
        count = df[col].sum()
        percentage = count / len(df) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {col.upper():11s}: {count:>6,}개 ({percentage:>5.1f}%) {bar}")

    # 멀티 플랫폼 분석
    df['platform_count'] = df[available_platforms].sum(axis=1)
    print("\n플랫폼 지원 개수별 게임 분포:")
    for count in sorted(df['platform_count'].unique()):
        games = (df['platform_count'] == count).sum()
        percentage = games / len(df) * 100
        print(f"  {int(count)}개 플랫폼 지원: {games:>6,}개 ({percentage:>5.1f}%)")

# ============================================================================
# 11. 출시 연도 분석
# ============================================================================
print("\n" + "=" * 80)
print("11. 출시 연도 분석")
print("=" * 80)
if 'date_release' in df.columns:
    df_temp = df.copy()
    df_temp['date_release'] = pd.to_datetime(df_temp['date_release'], errors='coerce')
    df_temp['year'] = df_temp['date_release'].dt.year

    valid_years = df_temp['year'].notna().sum()
    print(f"\n유효한 날짜 데이터: {valid_years:,}개 ({valid_years/len(df)*100:.1f}%)")

    if valid_years > 0:
        print(f"최초 출시: {int(df_temp['year'].min())}년")
        print(f"최근 출시: {int(df_temp['year'].max())}년")

        # 연도별 통계
        year_counts = df_temp['year'].value_counts().sort_index()
        print(f"\n연도별 게임 출시 통계:")
        print(f"  평균 연간 출시: {year_counts.mean():.1f}개")

        # Top 5 출시 연도
        print(f"\nTop 5 가장 많은 게임이 출시된 연도:")
        for year, count in year_counts.nlargest(5).items():
            print(f"  {int(year)}년: {count:>6,}개")

# ============================================================================
# 12. 시각화 생성
# ============================================================================
print("\n" + "=" * 80)
print("12. 시각화 생성")
print("=" * 80)

output_dir = os.path.join(project_root, 'results', 'eda_games_visualizations')
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Rating 분포
if 'rating' in df.columns:
    plt.figure(figsize=(12, 6))
    rating_counts = df['rating'].value_counts()
    colors = sns.color_palette("RdYlGn_r", len(rating_counts))
    plt.barh(range(len(rating_counts)), rating_counts.values, color=colors, alpha=0.8)
    plt.yticks(range(len(rating_counts)), rating_counts.index, fontsize=10)
    plt.xlabel('Number of Games', fontsize=12, fontweight='bold')
    plt.title('Game Rating Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    for i, v in enumerate(rating_counts.values):
        plt.text(v + max(rating_counts.values)*0.01, i, f'{v:,}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 1. Rating 분포 저장 완료")

# 2. Positive Ratio 분포
if 'positive_ratio' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.hist(df['positive_ratio'], bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(df['positive_ratio'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'평균: {df["positive_ratio"].mean():.1f}%')
    plt.axvline(df['positive_ratio'].median(), color='blue', linestyle='--', linewidth=2,
                label=f'중간값: {df["positive_ratio"].median():.1f}%')
    plt.xlabel('Positive Ratio (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games', fontsize=12, fontweight='bold')
    plt.title('Distribution of Positive Review Ratio', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_positive_ratio_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 2. Positive Ratio 분포 저장 완료")

# 3. User Reviews 분포 (로그 스케일)
if 'user_reviews' in df.columns:
    plt.figure(figsize=(12, 6))
    user_reviews_clean = df['user_reviews'][df['user_reviews'] > 0]
    plt.hist(np.log10(user_reviews_clean), bins=50, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Log10(User Reviews)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games', fontsize=12, fontweight='bold')
    plt.title('Distribution of User Reviews (Log Scale)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_user_reviews_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 3. User Reviews 분포 저장 완료")

# 4. Price 분포
if 'price_final' in df.columns:
    plt.figure(figsize=(12, 6))
    price_clean = df['price_final'][df['price_final'] > 0]
    plt.hist(price_clean, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(price_clean.mean(), color='red', linestyle='--', linewidth=2,
                label=f'평균: ${price_clean.mean():.2f}')
    plt.axvline(price_clean.median(), color='blue', linestyle='--', linewidth=2,
                label=f'중간값: ${price_clean.median():.2f}')
    plt.xlabel('Price ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games', fontsize=12, fontweight='bold')
    plt.title('Distribution of Game Prices (Paid Games Only)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 4. Price 분포 저장 완료")

# 5. Platform 지원
if available_platforms:
    plt.figure(figsize=(10, 6))
    platform_counts = [df[col].sum() for col in available_platforms]
    platform_names = [col.upper() for col in available_platforms]
    colors_platform = ['#0078D4', '#999999', '#FCC624', '#1A1A1A']
    bars = plt.bar(platform_names, platform_counts, color=colors_platform[:len(platform_names)], alpha=0.8)
    plt.ylabel('Number of Games', fontsize=12, fontweight='bold')
    plt.title('Games by Platform', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for bar, count in zip(bars, platform_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(platform_counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_platform_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 5. Platform 분포 저장 완료")

# 6. Release Year 트렌드
if 'date_release' in df.columns:
    df_temp = df.copy()
    df_temp['date_release'] = pd.to_datetime(df_temp['date_release'], errors='coerce')
    df_temp['year'] = df_temp['date_release'].dt.year
    year_counts = df_temp['year'].value_counts().sort_index()
    year_counts = year_counts[(year_counts.index >= 2000) & (year_counts.index <= 2024)]

    plt.figure(figsize=(14, 6))
    plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2.5,
             markersize=7, color='darkblue', markerfacecolor='lightblue', markeredgewidth=2)
    plt.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='lightblue')
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games Released', fontsize=12, fontweight='bold')
    plt.title('Game Releases by Year (2000-2024)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(2000, 2025, 2))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_releases_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 6. Release Year 트렌드 저장 완료")

print(f"\n모든 시각화 완료! 저장 위치: {output_dir}/")

print("\n" + "=" * 80)
print("games.csv EDA 완료!")
print("=" * 80)
