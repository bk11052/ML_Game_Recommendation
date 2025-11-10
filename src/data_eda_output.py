"""
output.csv 전용 EDA 스크립트
리뷰 데이터 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
print("=" * 80)
print("OUTPUT.CSV - 리뷰 데이터 분석")
print("=" * 80)
print("\n데이터 로드 중...")

# 스크립트 위치에 따라 경로 자동 조정
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data', 'output.csv')

df = pd.read_csv(data_path)
print("✓ 데이터 로드 완료\n")

# ============================================================================
# 1. 기본 정보
# ============================================================================
print("=" * 80)
print("1. 기본 정보")
print("=" * 80)
print(f"총 리뷰 수: {len(df):,}개")
print(f"컬럼 수: {len(df.columns)}개")
print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n컬럼 목록:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

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
    print("✓ 결측치 없음!")
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
# 6. 긍정/부정 리뷰 분석
# ============================================================================
print("\n" + "=" * 80)
print("6. 긍정/부정 리뷰 분석")
print("=" * 80)
if 'is_positive' in df.columns:
    sentiment_counts = df['is_positive'].value_counts()
    print("\n리뷰 감정 분포:")
    for sentiment, count in sentiment_counts.items():
        percentage = count / len(df) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {sentiment:10s}: {count:>8,}개 ({percentage:>5.1f}%) {bar}")

    # 균형도 체크
    ratio = sentiment_counts.max() / sentiment_counts.min()
    print(f"\n데이터 균형도: {ratio:.2f}:1", end="")
    if ratio < 1.2:
        print(" (매우 균형잡힌 데이터셋! ✓)")
    elif ratio < 2.0:
        print(" (균형잡힌 데이터셋)")
    else:
        print(" (불균형 데이터셋)")

# ============================================================================
# 7. 게임별 리뷰 분석
# ============================================================================
print("\n" + "=" * 80)
print("7. 게임별 리뷰 분석")
print("=" * 80)
if 'app_id' in df.columns:
    unique_games = df['app_id'].nunique()
    print(f"\n리뷰가 있는 고유 게임 수: {unique_games:,}개")
    print(f"게임당 평균 리뷰 수: {len(df) / unique_games:.2f}개")

    # 게임별 리뷰 수 분포
    reviews_per_game = df['app_id'].value_counts()
    print(f"\n게임별 리뷰 수 통계:")
    print(f"  최소: {reviews_per_game.min():,}개")
    print(f"  최대: {reviews_per_game.max():,}개")
    print(f"  평균: {reviews_per_game.mean():.1f}개")
    print(f"  중간값: {reviews_per_game.median():.1f}개")
    print(f"  표준편차: {reviews_per_game.std():.1f}개")

    # Top 20 리뷰 많은 게임
    print(f"\n📌 Top 20 리뷰가 많은 게임 (app_id):")
    top_games = reviews_per_game.head(20)
    for i, (app_id, count) in enumerate(top_games.items(), 1):
        # 해당 게임의 긍정/부정 비율
        game_reviews = df[df['app_id'] == app_id]
        if 'is_positive' in df.columns:
            positive = (game_reviews['is_positive'] == 'Positive').sum()
            negative = (game_reviews['is_positive'] == 'Negative').sum()
            pos_ratio = positive / (positive + negative) * 100 if (positive + negative) > 0 else 0
            print(f"  {i:2d}. app_id {app_id:>6}: {count:>6,}개 리뷰 | 긍정 {pos_ratio:>5.1f}% ({positive:>6,}개) | 부정 {100-pos_ratio:>5.1f}% ({negative:>6,}개)")
        else:
            print(f"  {i:2d}. app_id {app_id:>6}: {count:>6,}개 리뷰")

# ============================================================================
# 8. 리뷰어 분석
# ============================================================================
print("\n" + "=" * 80)
print("8. 리뷰어(Author) 분석")
print("=" * 80)
if 'author_id' in df.columns:
    unique_authors = df['author_id'].nunique()
    print(f"\n고유 리뷰어 수: {unique_authors:,}명")
    print(f"리뷰어당 평균 리뷰 수: {len(df) / unique_authors:.2f}개")

    # 리뷰어별 리뷰 수 분포
    reviews_per_author = df['author_id'].value_counts()
    print(f"\n리뷰어별 리뷰 수 통계:")
    print(f"  최소: {reviews_per_author.min()}개")
    print(f"  최대: {reviews_per_author.max()}개")
    print(f"  평균: {reviews_per_author.mean():.2f}개")
    print(f"  중간값: {reviews_per_author.median():.1f}개")

    # 리뷰 활동도 분포
    print(f"\n리뷰 활동도 분포:")
    one_review = (reviews_per_author == 1).sum()
    two_five = ((reviews_per_author >= 2) & (reviews_per_author <= 5)).sum()
    six_ten = ((reviews_per_author >= 6) & (reviews_per_author <= 10)).sum()
    more_ten = (reviews_per_author > 10).sum()

    print(f"  1개 리뷰만 작성:  {one_review:>6,}명 ({one_review/unique_authors*100:>5.1f}%)")
    print(f"  2-5개 리뷰:       {two_five:>6,}명 ({two_five/unique_authors*100:>5.1f}%)")
    print(f"  6-10개 리뷰:      {six_ten:>6,}명 ({six_ten/unique_authors*100:>5.1f}%)")
    print(f"  10개 이상 리뷰:   {more_ten:>6,}명 ({more_ten/unique_authors*100:>5.1f}%)")

    # Top 10 가장 활발한 리뷰어
    print(f"\n📌 Top 10 가장 활발한 리뷰어:")
    top_authors = reviews_per_author.head(10)
    for i, (author_id, count) in enumerate(top_authors.items(), 1):
        print(f"  {i:2d}. author_id {author_id}: {count:>3}개 리뷰")

# ============================================================================
# 9. 리뷰 텍스트 분석
# ============================================================================
print("\n" + "=" * 80)
print("9. 리뷰 텍스트(Content) 분석")
print("=" * 80)
if 'content' in df.columns:
    valid_content = df['content'].dropna()
    print(f"\n유효한 리뷰 텍스트: {len(valid_content):,}개 ({len(valid_content)/len(df)*100:.1f}%)")

    # 리뷰 길이 분석
    content_lengths = valid_content.str.len()
    print(f"\n리뷰 텍스트 길이 통계:")
    print(f"  평균: {content_lengths.mean():.1f}자")
    print(f"  중간값: {content_lengths.median():.1f}자")
    print(f"  최소: {content_lengths.min()}자")
    print(f"  최대: {content_lengths.max():,}자")
    print(f"  표준편차: {content_lengths.std():.1f}자")

    # 길이별 분포
    very_short = (content_lengths < 10).sum()
    short = ((content_lengths >= 10) & (content_lengths < 50)).sum()
    medium = ((content_lengths >= 50) & (content_lengths < 200)).sum()
    long_text = ((content_lengths >= 200) & (content_lengths < 1000)).sum()
    very_long = (content_lengths >= 1000).sum()
    total = len(content_lengths)

    print(f"\n리뷰 길이 분포:")
    print(f"  매우 짧음 (<10자):     {very_short:>8,}개 ({very_short/total*100:>5.1f}%)")
    print(f"  짧음 (10-50자):        {short:>8,}개 ({short/total*100:>5.1f}%)")
    print(f"  보통 (50-200자):       {medium:>8,}개 ({medium/total*100:>5.1f}%)")
    print(f"  긴 리뷰 (200-1000자):  {long_text:>8,}개 ({long_text/total*100:>5.1f}%)")
    print(f"  매우 긴 리뷰 (1000자+): {very_long:>8,}개 ({very_long/total*100:>5.1f}%)")

    # 가장 짧은/긴 리뷰
    shortest_idx = content_lengths.idxmin()
    longest_idx = content_lengths.idxmax()

    print(f"\n📝 가장 짧은 리뷰 ({content_lengths.min()}자):")
    print(f"   \"{valid_content.loc[shortest_idx][:100]}\"")

    print(f"\n📝 가장 긴 리뷰 ({content_lengths.max():,}자):")
    print(f"   \"{valid_content.loc[longest_idx][:200]}...\"")

    # 공통 단어 분석 (상위 20개)
    print(f"\n📊 가장 자주 사용된 단어 Top 20:")
    # 모든 텍스트를 하나로 합치고 단어 분리
    all_text = ' '.join(valid_content.astype(str).str.lower())
    words = all_text.split()
    # 너무 짧은 단어 제외 (2자 이하)
    words = [w for w in words if len(w) > 2]
    word_counts = Counter(words).most_common(20)
    for i, (word, count) in enumerate(word_counts, 1):
        print(f"   {i:2d}. '{word}': {count:,}회")

# ============================================================================
# 10. 리뷰 ID 분석
# ============================================================================
print("\n" + "=" * 80)
print("10. 리뷰 ID 분석")
print("=" * 80)
if 'id' in df.columns:
    print(f"\n고유 리뷰 ID 수: {df['id'].nunique():,}개")
    print(f"전체 행 수: {len(df):,}개")

    duplicates = len(df) - df['id'].nunique()
    if duplicates == 0:
        print("✓ 중복 없음 - 모든 리뷰 ID가 고유합니다!")
    else:
        print(f"⚠️  중복 리뷰 ID: {duplicates}개")

    print(f"\nID 범위:")
    print(f"  최소 ID: {df['id'].min()}")
    print(f"  최대 ID: {df['id'].max()}")

# ============================================================================
# 11. 게임별 긍정/부정 비율
# ============================================================================
print("\n" + "=" * 80)
print("11. 게임별 긍정/부정 비율 분석")
print("=" * 80)
if 'app_id' in df.columns and 'is_positive' in df.columns:
    # 게임별 긍정 비율 계산
    game_sentiment = df.groupby('app_id')['is_positive'].apply(
        lambda x: (x == 'Positive').sum() / len(x) * 100
    ).sort_values(ascending=False)

    print(f"\n게임별 긍정 비율 통계:")
    print(f"  평균: {game_sentiment.mean():.1f}%")
    print(f"  중간값: {game_sentiment.median():.1f}%")
    print(f"  최소: {game_sentiment.min():.1f}%")
    print(f"  최대: {game_sentiment.max():.1f}%")

    # 가장 긍정적인 게임 Top 10
    print(f"\n📌 가장 긍정적인 평가를 받은 게임 Top 10 (app_id):")
    for i, (app_id, ratio) in enumerate(game_sentiment.head(10).items(), 1):
        count = len(df[df['app_id'] == app_id])
        print(f"  {i:2d}. app_id {app_id}: 긍정률 {ratio:>5.1f}% ({count:>5,}개 리뷰)")

    # 가장 부정적인 게임 Top 10
    print(f"\n📌 가장 부정적인 평가를 받은 게임 Top 10 (app_id):")
    for i, (app_id, ratio) in enumerate(game_sentiment.tail(10).items(), 1):
        count = len(df[df['app_id'] == app_id])
        print(f"  {i:2d}. app_id {app_id}: 긍정률 {ratio:>5.1f}% ({count:>5,}개 리뷰)")

# ============================================================================
# 12. 시각화 생성
# ============================================================================
print("\n" + "=" * 80)
print("12. 시각화 생성")
print("=" * 80)

output_dir = os.path.join(project_root, 'results', 'eda_output_visualizations')
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. 긍정/부정 리뷰 분포
if 'is_positive' in df.columns:
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['is_positive'].value_counts()
    colors = ['#2ecc71' if 'Positive' in str(x) else '#e74c3c' for x in sentiment_counts.index]
    bars = plt.bar(range(len(sentiment_counts)), sentiment_counts.values,
                   color=colors, alpha=0.8, width=0.6)
    plt.xticks(range(len(sentiment_counts)), sentiment_counts.index, fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12, fontweight='bold')
    plt.title('Review Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        plt.text(i, count + max(sentiment_counts.values)*0.02,
                f'{count:,}\n({count/len(df)*100:.1f}%)',
                ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 1. Sentiment 분포 저장 완료")

# 2. 게임별 리뷰 수 분포
if 'app_id' in df.columns:
    plt.figure(figsize=(12, 6))
    reviews_per_game = df['app_id'].value_counts()
    plt.hist(reviews_per_game.values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Reviews per Game', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games', fontsize=12, fontweight='bold')
    plt.title('Distribution of Reviews per Game', fontsize=14, fontweight='bold')
    plt.axvline(reviews_per_game.mean(), color='red', linestyle='--', linewidth=2,
                label=f'평균: {reviews_per_game.mean():.1f}')
    plt.axvline(reviews_per_game.median(), color='green', linestyle='--', linewidth=2,
                label=f'중간값: {reviews_per_game.median():.1f}')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_reviews_per_game_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 2. 게임별 리뷰 수 분포 저장 완료")

# 3. Top 20 리뷰 많은 게임
if 'app_id' in df.columns:
    plt.figure(figsize=(12, 8))
    reviews_per_game = df['app_id'].value_counts().head(20)
    colors_gradient = sns.color_palette("YlOrRd", len(reviews_per_game))
    plt.barh(range(len(reviews_per_game)), reviews_per_game.values,
             color=colors_gradient, alpha=0.8)
    plt.yticks(range(len(reviews_per_game)),
               [f'app_id {x}' for x in reviews_per_game.index], fontsize=10)
    plt.xlabel('Number of Reviews', fontsize=12, fontweight='bold')
    plt.title('Top 20 Most Reviewed Games', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()
    for i, v in enumerate(reviews_per_game.values):
        plt.text(v + max(reviews_per_game.values)*0.01, i, f'{v:,}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_top_20_games.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 3. Top 20 게임 저장 완료")

# 4. 리뷰 길이 분포
if 'content' in df.columns:
    plt.figure(figsize=(12, 6))
    content_lengths = df['content'].dropna().str.len()
    # 이상치 제거 (99 percentile)
    upper_limit = content_lengths.quantile(0.99)
    content_lengths_filtered = content_lengths[content_lengths <= upper_limit]

    plt.hist(content_lengths_filtered, bins=50, color='mediumseagreen', alpha=0.7, edgecolor='black')
    plt.xlabel('Review Length (characters)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Reviews', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of Review Length (up to {upper_limit:.0f} chars, 99th percentile)',
              fontsize=14, fontweight='bold')
    plt.axvline(content_lengths.mean(), color='red', linestyle='--', linewidth=2,
                label=f'평균: {content_lengths.mean():.1f}')
    plt.axvline(content_lengths.median(), color='blue', linestyle='--', linewidth=2,
                label=f'중간값: {content_lengths.median():.1f}')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_review_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 4. 리뷰 길이 분포 저장 완료")

# 5. 리뷰어 활동도
if 'author_id' in df.columns:
    plt.figure(figsize=(10, 6))
    reviews_per_author = df['author_id'].value_counts()

    categories = ['1개', '2-5개', '6-10개', '10개+']
    counts = [
        (reviews_per_author == 1).sum(),
        ((reviews_per_author >= 2) & (reviews_per_author <= 5)).sum(),
        ((reviews_per_author >= 6) & (reviews_per_author <= 10)).sum(),
        (reviews_per_author > 10).sum()
    ]

    colors_activity = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = plt.bar(categories, counts, color=colors_activity, alpha=0.8)
    plt.ylabel('Number of Reviewers', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Reviews Written', fontsize=12, fontweight='bold')
    plt.title('Reviewer Activity Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_reviewer_activity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 5. 리뷰어 활동도 저장 완료")

# 6. 게임별 긍정 비율
if 'app_id' in df.columns and 'is_positive' in df.columns:
    plt.figure(figsize=(12, 6))
    game_sentiment = df.groupby('app_id')['is_positive'].apply(
        lambda x: (x == 'Positive').sum() / len(x) * 100
    )

    plt.hist(game_sentiment.values, bins=30, color='coral', alpha=0.7, edgecolor='black')
    plt.xlabel('Positive Review Ratio (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games', fontsize=12, fontweight='bold')
    plt.title('Distribution of Positive Review Ratio by Game', fontsize=14, fontweight='bold')
    plt.axvline(game_sentiment.mean(), color='red', linestyle='--', linewidth=2,
                label=f'평균: {game_sentiment.mean():.1f}%')
    plt.axvline(game_sentiment.median(), color='blue', linestyle='--', linewidth=2,
                label=f'중간값: {game_sentiment.median():.1f}%')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_positive_ratio_by_game.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 6. 게임별 긍정 비율 저장 완료")

print(f"\n모든 시각화 완료! 저장 위치: {output_dir}/")

print("\n" + "=" * 80)
print("output.csv EDA 완료!")
print("=" * 80)
