# 🎮 Steam Game Recommendation System - PPT 발표 자료

---

## 1️⃣ End-to-End Process

### 프로젝트 워크플로우
```
1. Business Objective 설정
   ↓
2. Data Collection & Integration
   - games.csv (50,872 games) + output.csv (443,144 reviews)
   - Merged → merged_data.csv (250,423 rows)
   ↓
3. Exploratory Data Analysis (EDA)
   - 10가지 시각화 분석
   - 결측치 분석 (20% 리뷰 데이터 결측)
   ↓
4. Data Preprocessing
   - 중복 제거, 결측치 처리
   - 데이터 타입 변환 (날짜, bool, 숫자)
   - 인코딩 (is_positive, rating)
   - 이상치 제거
   - MinMax 정규화
   ↓
5. Model Training (6가지 알고리즘)
   - CF Baseline
   - CBF Baseline
   - Hybrid Simple (Weighted)
   - Hybrid Stacking (ML-based)
   - Factorization Machines
   - Word Embeddings
   ↓
6. Model Evaluation
   - Precision@K, Recall@K, F1@K
   - NDCG@K
   - Coverage, Running Time
   ↓
7. Model Comparison & Selection
```

---

## 2️⃣ Business Objective

### 프로젝트 목표
**Steam 게임 추천 시스템 구축 및 알고리즘 비교 분석**

### 핵심 질문 (Research Questions)
1. **어떤 추천 알고리즘이 Steam 게임 데이터에 가장 효과적인가?**
2. **Collaborative Filtering vs Content-Based Filtering - 어느 접근이 더 우수한가?**
3. **Hybrid 방식이 단일 모델보다 성능이 우수한가?**
4. **고급 기법(FM, Word2Vec)이 전통적 방법보다 나은가?**

### 비즈니스 가치
- **사용자 경험 향상**: 개인화된 게임 추천으로 만족도 증가
- **플랫폼 수익 증대**: 적절한 추천으로 게임 구매 전환율 향상
- **Long-tail 게임 발굴**: 인기 게임 외 숨겨진 명작 추천

### 성공 지표
- Precision@10 > 0.15
- Recall@10 > 0.10
- 실행 시간 < 60초 (실용성)

---

## 3️⃣ Data Exploration

### 📊 데이터셋 개요

| 구분 | 내용 |
|------|------|
| **총 게임 수** | 50,872개 |
| **총 리뷰 수** | 443,144개 |
| **병합 데이터** | 250,423 rows × 17 columns |
| **기간** | 2000년 ~ 2024년 |

### 📈 주요 컬럼

#### 리뷰 데이터 (output.csv)
- `id`: 리뷰 ID
- `app_id`: 게임 ID (병합 키)
- `content`: 리뷰 텍스트
- `author_id`: 작성자 ID
- `is_positive`: 긍정/부정 (Positive/Negative)

#### 게임 메타데이터 (games.csv)
- `title`: 게임 제목
- `date_release`: 출시일
- `win`, `mac`, `linux`, `steam_deck`: 플랫폼 지원
- `rating`: 평점 (Overwhelmingly Positive ~ Negative)
- `positive_ratio`: 긍정 리뷰 비율 (0~100)
- `user_reviews`: 총 리뷰 수
- `price_final`, `price_original`, `discount`: 가격 정보

### 🔍 EDA 주요 발견사항

#### 1. 결측치 분석
- **리뷰 데이터 결측**: 20.29% (50,823개 게임)
  - `id`, `content`, `author_id`, `is_positive` 동시 결측
  - **원인**: 리뷰가 없는 신작 게임 또는 비인기 게임
- **게임 메타데이터**: 결측 없음 (완전한 게임 정보)

#### 2. 긍정/부정 리뷰 분포
- **Positive**: 101,231개 (50.7%)
- **Negative**: 98,369개 (49.3%)
- **균형잡힌 데이터셋** → 편향 없는 학습 가능

#### 3. 게임 평점 분포
- **Very Positive**: 133,485개 (53.3%)
- **Overwhelmingly Positive**: 66,287개 (26.5%)
- **Mixed 이하**: 14,757개 (5.9%)
- → Steam 게임 대부분이 긍정적 평가

#### 4. 인기 게임 Top 10
1. Counter-Strike: Global Offensive (24,802 리뷰)
2. Dota 2 (20,001 리뷰)
3. Team Fortress 2 (20,000 리뷰)
4. Left 4 Dead 2 (15,321 리뷰)
5. Counter-Strike: Source (12,409 리뷰)

→ **Valve 게임 독점** (FPS, MOBA 장르 강세)

#### 5. 가격 분포
- **무료 게임**: 다수 포함
- **유료 게임 중간값**: $0.99
- **최고가**: $299.99
- → 저가 인디 게임이 대부분

#### 6. 플랫폼 지원
- **Windows**: 대부분
- **Mac**: 일부 지원
- **Linux**: 소수 지원
- **Steam Deck**: 최근 추가된 플랫폼

#### 7. 출시 연도 트렌드
- **2007년 급증**: Steam 본격 확장 시기
- **2010년대 중반 정점**: 인디 게임 붐
- **최근 완만**: 시장 성숙기

### 📊 시각화 자료 (10개)
1. **Missing Values Heatmap**: 결측치 패턴 확인
2. **Rating Distribution**: 평점 분포
3. **Positive Ratio Distribution**: 긍정 비율 분포
4. **User Reviews Distribution**: 리뷰 수 분포 (로그 스케일)
5. **Price Distribution**: 가격 분포
6. **Platform Distribution**: 플랫폼별 게임 수
7. **Releases by Year**: 연도별 출시 게임 트렌드
8. **Top 20 Games**: 가장 많이 리뷰된 게임
9. **Ratio vs Reviews Scatter**: 긍정 비율 vs 리뷰 수 관계
10. **Sentiment Distribution**: 긍정/부정 리뷰 분포

---

## 4️⃣ Data Preprocessing

### 전처리 파이프라인

#### Step 1: 결측치 및 중복 데이터 처리
```python
# 중복 제거 (리뷰 ID 기준)
preprocessed_df.drop_duplicates(subset=['id'], inplace=True)

# 필수 컬럼 결측치 제거 (리뷰 텍스트)
preprocessed_df.dropna(subset=['content'], inplace=True)

# 숫자형 컬럼 결측치 0으로 채우기
num_cols = ['positive_ratio', 'user_reviews', 'price_final', 'price_original', 'discount']
preprocessed_df[num_cols] = preprocessed_df[num_cols].fillna(0)
```

#### Step 2: 데이터 타입 변환
```python
# 날짜 타입
preprocessed_df['date_release'] = pd.to_datetime(preprocessed_df['date_release'], errors='coerce')

# 불리언 타입
bool_cols = ['win', 'mac', 'linux', 'steam_deck']
preprocessed_df[bool_cols] = preprocessed_df[bool_cols].map({'True': True, 'False': False})

# 정수 타입
preprocessed_df['user_reviews'] = preprocessed_df['user_reviews'].astype(int)
```

#### Step 3: 인코딩
```python
# 이진 인코딩 (is_positive)
preprocessed_df['is_positive_encoded'] = preprocessed_df['is_positive'].map({
    'Positive': 1,
    'Negative': 0
})

# 원-핫 인코딩 (rating)
preprocessed_df = pd.get_dummies(preprocessed_df, columns=['rating'], prefix='rating')
```

#### Step 4: 이상치 제거
```python
# positive_ratio: 0~100 범위 검증
preprocessed_df = preprocessed_df[
    (preprocessed_df['positive_ratio'] >= 0) &
    (preprocessed_df['positive_ratio'] <= 100)
]

# price_final: 음수 제거
preprocessed_df = preprocessed_df[preprocessed_df['price_final'] >= 0]
```

#### Step 5: 정규화/스케일링
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cols_to_scale = ['user_reviews', 'price_final', 'positive_ratio']
preprocessed_df[cols_to_scale] = scaler.fit_transform(preprocessed_df[cols_to_scale])
```

### 전처리 결과
- **처리 전**: 250,423 rows
- **처리 후**: 199,178 rows (리뷰 텍스트 있는 데이터만)
- **제거된 데이터**: 51,245 rows (리뷰 없는 게임)

---

## 5️⃣ Modeling

### 구현된 6가지 추천 알고리즘

#### 1. CF Baseline (Collaborative Filtering)
**기법**: Matrix Factorization with SGD

```python
# 핵심 아이디어
User-Item Interaction Matrix → Latent Factors (k=50)
P(user factors) × Q(item factors) = Predicted Rating
```

**특징**:
- 유저-게임 상호작용 기반
- Cold-start 문제 존재
- 유사 취향 유저 기반 추천

---

#### 2. CBF Baseline (Content-Based Filtering)
**기법**: TF-IDF + Cosine Similarity

```python
# 핵심 아이디어
Game Title → TF-IDF Vector → Cosine Similarity
Similar Games → Recommendation
```

**특징**:
- 게임 메타데이터(제목) 기반
- Cold-start 문제 해결 가능
- 유사한 게임만 추천 (다양성 부족)

---

#### 3. Hybrid Simple (Weighted Combination)
**기법**: 가중치 결합 (CF 30% + CBF 70%)

```python
# 핵심 아이디어
Final_Score = 0.3 × CF_Score + 0.7 × CBF_Score
```

**특징**:
- 단순하지만 효과적
- 두 접근법의 장점 결합
- 가중치 튜닝 필요

---

#### 4. Hybrid Stacking ⭐
**기법**: LogisticRegression 메타 모델

```python
# 핵심 아이디어
CF_Score + CBF_Score → LogisticRegression → Final Prediction
Meta-learner가 최적 가중치 자동 학습
```

**특징**:
- 학습 기반 결합 (데이터 기반 최적화)
- CF와 CBF의 예측을 메타 피처로 사용
- 일반화 성능 우수

---

#### 5. Factorization Machines
**기법**: 2차 피처 상호작용 학습

```python
# 핵심 아이디어
y = w0 + Σ(wi·xi) + Σ(Σ(<vi,vj>·xi·xj))
피처 조합을 통한 고급 예측
```

**특징**:
- 게임 메타데이터 활용
- 희소 데이터 처리 우수
- 피처 간 상호작용 모델링

---

#### 6. Word Embeddings
**기법**: Word2Vec (Skip-gram)

```python
# 핵심 아이디어
Game Title → Word2Vec Embedding → Semantic Similarity
의미적으로 유사한 게임 추천
```

**특징**:
- 의미적 유사도 학습
- "RPG", "Adventure" 등 장르 단어 캡처
- 제목이 비슷하지 않아도 유사 게임 발견

---

### 모델 비교 요약

| 모델 | 접근법 | 장점 | 단점 |
|------|--------|------|------|
| **CF Baseline** | 협업 필터링 | 집단 지성 활용 | Cold-start |
| **CBF Baseline** | 콘텐츠 기반 | Cold-start 해결 | 다양성 부족 |
| **Hybrid Simple** | 가중치 결합 | 단순하고 효과적 | 수동 튜닝 필요 |
| **Hybrid Stacking** | 학습 기반 결합 | 자동 최적화 | 학습 시간 증가 |
| **FM** | 피처 상호작용 | 메타데이터 활용 | 복잡도 높음 |
| **Word2Vec** | 의미 임베딩 | 의미적 유사도 | 제목 의존적 |

---

## 6️⃣ Learning Model Evaluation and Analysis

### 평가 지표

#### Precision@K
```
Precision@K = (추천된 K개 중 실제 좋아한 게임 수) / K
```
- **의미**: 추천의 정확도
- **높을수록 좋음**: 추천이 정확함

#### Recall@K
```
Recall@K = (추천된 K개 중 실제 좋아한 게임 수) / (전체 좋아한 게임 수)
```
- **의미**: 추천의 재현율
- **높을수록 좋음**: 좋아할 게임을 많이 찾아냄

#### F1@K
```
F1@K = 2 × (Precision@K × Recall@K) / (Precision@K + Recall@K)
```
- **의미**: Precision과 Recall의 조화 평균
- **균형잡힌 지표**

#### NDCG@K (Normalized Discounted Cumulative Gain)
- **의미**: 순위를 고려한 평가
- **상위 추천의 정확도에 가중치**

#### Coverage
```
Coverage = (추천된 unique 게임 수) / (전체 게임 수)
```
- **의미**: 추천 다양성
- **높을수록 다양한 게임 추천**

#### Running Time
- **의미**: 모델 실행 속도
- **실용성 평가**

---

### 🔬 예상 결과 분석

#### 시나리오 1: CF Baseline이 가장 우수
**예상 원인**:
- Steam 리뷰 데이터는 유저-게임 상호작용이 풍부
- 협업 필터링이 집단 지성을 잘 활용
- 유사 취향 유저 기반 추천이 효과적

**시사점**:
- 리뷰 데이터 품질이 중요
- 유저 행동 패턴이 강한 신호

---

#### 시나리오 2: Hybrid Stacking이 가장 우수
**예상 원인**:
- CF의 집단 지성 + CBF의 메타데이터 정보 결합
- 메타 러너가 최적 가중치 자동 학습
- 두 접근법의 약점 보완

**시사점**:
- 앙상블 기법의 우수성
- 학습 기반 결합이 수동 가중치보다 효과적

---

#### 시나리오 3: Word2Vec이 의외로 우수
**예상 원인**:
- 게임 제목에 장르/특징 정보 풍부
- 의미적 유사도가 유저 취향과 일치
- "RPG", "Shooter" 등 키워드가 강한 신호

**시사점**:
- 텍스트 임베딩의 잠재력
- 제목만으로도 효과적인 추천 가능

---

#### 시나리오 4: 모든 모델 성능이 낮음 (실패 케이스)
**가능한 원인**:
1. **데이터 품질 문제**
   - 결측치 20%가 성능에 악영향
   - 리뷰가 없는 게임이 너무 많음

2. **평가 방법 문제**
   - Ground truth 설정이 부적절
   - Train/Test split이 불균형

3. **모델 설정 문제**
   - 하이퍼파라미터 튜닝 부족
   - K 값 설정이 부적절

**해결 방안**:
- 리뷰가 충분한 게임만 필터링 (user_reviews > 10)
- 교차 검증 적용
- 그리드 서치로 하이퍼파라미터 최적화

---

### 📊 결과 시각화 계획
1. **모델 비교 막대 그래프**: Precision@K, Recall@K, F1@K
2. **실행 시간 비교**: Running Time
3. **Coverage 비교**: 추천 다양성
4. **K값에 따른 성능 변화**: K=5, 10, 20

---

## 7️⃣ Learning Experience

### 💪 Difficulties Encountered and Solutions

#### 문제 1: Git 브랜치 관리 혼란
**어려움**:
- 여러 브랜치에 코드가 분산되어 있음
- 어떤 브랜치가 최신 코드인지 불명확
- 병합 시 충돌 발생

**해결 방법**:
- 각 브랜치의 목적과 내용을 체계적으로 분석
- 깔끔한 프로젝트 구조로 main 브랜치 재구성
- src/, models/, data/, results/ 디렉토리 분리

**배운 점**:
- 브랜치 전략의 중요성 (feature branch 사용)
- 초기에 프로젝트 구조 설계 필요
- Git workflow 이해 향상

---

#### 문제 2: 데이터 병합 전략 선택
**어려움**:
- Left join vs Right join 선택 고민
- 처음에는 리뷰 있는 게임만 포함 (left join)
- 게임 수가 50,872 → 50으로 급감

**해결 방법**:
- Right join으로 변경 → 모든 게임 포함
- 리뷰 없는 게임은 NaN으로 유지
- 결측치는 전처리 단계에서 처리

**배운 점**:
- 데이터 병합 전략이 전체 프로젝트에 미치는 영향
- 비즈니스 목표에 맞는 데이터 선택 중요
- 결측치 처리 전략의 다양성

---

#### 문제 3: 결측치 표현 불일치
**어려움**:
- pandas info()에서 non-null로 표시되지만 실제로는 빈 문자열('')
- 결측치가 제대로 감지되지 않음

**해결 방법**:
```python
df = pd.read_csv('merged_data.csv',
                 na_values=['', ' ', 'NA', 'N/A', 'null', 'None'])
```
- `na_values` 파라미터로 빈 문자열을 NaN으로 처리

**배운 점**:
- 데이터 품질 검증의 중요성
- 겉보기와 실제 데이터 상태가 다를 수 있음
- pandas의 다양한 옵션 활용법

---

#### 문제 4: 파일 경로 문제
**어려움**:
- src/ 디렉토리에서 실행 시 `FileNotFoundError`
- 상대 경로 설정 오류

**해결 방법**:
```python
# Before
df = pd.read_csv('merged_data.csv')

# After
df = pd.read_csv('../data/merged_data.csv')
```
- 모든 스크립트의 상대 경로 수정

**배운 점**:
- 프로젝트 구조와 실행 위치 고려 필요
- 절대 경로보다 상대 경로가 이식성 좋음
- 테스트의 중요성

---

#### 문제 5: activelearning 모델의 메모리 문제
**어려움**:
- 코사인 유사도 행렬 2개 → 42GB 메모리 필요
- 개인 노트북 실행 불가능

**해결 방법 (고려)**:
1. On-demand 계산으로 메모리 28배 절감
2. Top-K 게임만 샘플링 (5,000개)
3. 해당 모델 제외

**배운 점**:
- 알고리즘의 공간 복잡도 중요성
- 실용성과 성능의 trade-off
- 최적화 기법의 필요성

---

### 🔄 If You Had More Time or Do It Again

#### 1. 데이터 수집 단계
**추가하고 싶은 것**:
- 더 많은 게임 메타데이터 (장르, 태그, 스크린샷)
- 유저 프로필 정보 (플레이 시간, 구매 이력)
- 시계열 데이터 (출시 후 평점 변화)

**이유**:
- 더 풍부한 피처로 모델 성능 향상
- CBF 모델의 다양성 증가
- 트렌드 분석 가능

---

#### 2. 전처리 단계
**다르게 하고 싶은 것**:
- 리뷰 텍스트 자연어 처리 (NLP)
  - 감성 분석 (Sentiment Analysis)
  - 주제 모델링 (Topic Modeling)
  - 키워드 추출
- 게임 태그 활용 (장르, 멀티플레이 등)

**이유**:
- 리뷰 content를 충분히 활용하지 못함
- 텍스트에 숨겨진 유용한 정보 많음

---

#### 3. 모델링 단계
**시도하고 싶은 것**:
- **딥러닝 모델**:
  - Neural Collaborative Filtering (NCF)
  - BERT 기반 리뷰 임베딩
  - Graph Neural Network (GNN)
- **앙상블 기법**:
  - Stacking 외에 Boosting, Bagging
  - 더 많은 base model 조합
- **하이퍼파라미터 튜닝**:
  - Grid Search, Random Search
  - Bayesian Optimization

**이유**:
- 최신 기법의 성능 비교
- 더 높은 정확도 달성 가능성

---

#### 4. 평가 단계
**개선하고 싶은 것**:
- **A/B 테스팅**: 실제 유저 반응 측정
- **온라인 평가**: 실시간 추천 성능
- **사용자 만족도 조사**: 정성적 평가
- **다양성 지표**: Intra-list Diversity, Serendipity

**이유**:
- Offline 평가만으로는 실제 효과 파악 어려움
- 비즈니스 임팩트 측정 필요

---

#### 5. 시스템 구현
**추가하고 싶은 것**:
- **웹 애플리케이션**: Flask/Django로 추천 시스템 배포
- **실시간 추천 API**: RESTful API 구현
- **대시보드**: 모델 성능 모니터링
- **데이터베이스**: SQLite/PostgreSQL 연동

**이유**:
- 실용적인 포트폴리오 구축
- 실제 서비스 경험

---

### 🎓 What I Have Learned (Individually)

#### Technical Skills
1. **추천 시스템 이론**
   - CF, CBF, Hybrid 방식의 원리와 장단점
   - Matrix Factorization, TF-IDF 수학적 이해
   - 평가 지표 (Precision, Recall, NDCG) 해석

2. **Python 라이브러리 활용**
   - pandas: 데이터 조작, 병합, 전처리
   - scikit-learn: 모델링, 평가
   - matplotlib/seaborn: 시각화
   - gensim: Word2Vec 임베딩

3. **Git/GitHub 협업**
   - 브랜치 관리 전략
   - Merge, Rebase, Force Push
   - Pull Request, Code Review

4. **프로젝트 구조 설계**
   - 모듈화, 디렉토리 구조
   - 재사용 가능한 코드 작성
   - Documentation (README, 주석)

---

#### Soft Skills
1. **문제 해결 능력**
   - 데이터 품질 문제 진단 및 해결
   - 메모리 최적화 전략 수립
   - 에러 디버깅 능력 향상

2. **의사 결정 능력**
   - 데이터 병합 전략 선택 (left vs right join)
   - 모델 선택 및 비교 전략 수립
   - Trade-off 고려 (성능 vs 복잡도)

3. **학습 능력**
   - 새로운 알고리즘 빠르게 학습
   - 공식 문서 읽고 적용
   - 오픈소스 코드 분석 및 활용

---

#### Domain Knowledge
1. **Steam 생태계 이해**
   - 게임 평가 시스템 (rating)
   - 유저 리뷰 특성
   - 플랫폼별 차이 (Win/Mac/Linux)

2. **추천 시스템 비즈니스**
   - Cold-start 문제의 실무적 중요성
   - 추천 다양성과 정확도의 균형
   - Long-tail 콘텐츠 발굴 가치

---

## 8️⃣ Teamwork Data

### 👥 Team Members
- **bk11052** (Team Lead)
- **Lim-K-M**
- **leejunggyun**
- **Gyu1026**

---

### 📋 Task Assignment

| Member | 담당 업무 | 주요 기여 |
|--------|----------|----------|
| **bk11052** | - 프로젝트 초기 설정<br>- GitHub 레포지토리 관리<br>- 데이터 수집 및 통합<br>- 최종 발표 자료 준비 | - Git 브랜치 전략 수립<br>- 데이터 병합 스크립트 작성<br>- 프로젝트 구조 재설계<br>- README 문서화 |
| **Lim-K-M** | - 데이터 전처리 파이프라인<br>- EDA 및 시각화<br>- 데이터 품질 검증 | - 전처리 스크립트 구현<br>- 10가지 시각화 생성<br>- 결측치 처리 전략 수립 |
| **leejunggyun** | - 추천 모델 구현 (6개)<br>- 알고리즘 연구 및 구현<br>- 성능 최적화 | - CF, CBF 베이스라인<br>- Hybrid, FM, Word2Vec 구현<br>- activelearning 모델 개발 |
| **Gyu1026** | - 평가 프레임워크 구축<br>- 모델 비교 분석<br>- 성능 지표 계산 | - run_eval_preprocessed.py<br>- Precision, Recall, NDCG 구현<br>- 결과 분석 및 시각화 |

---

### 📊 Contribution Percentage

| Member | 기여도 | 상세 |
|--------|--------|------|
| **bk11052** | 25% | 프로젝트 관리, 데이터 통합, 문서화 |
| **Lim-K-M** | 25% | 데이터 전처리, EDA, 시각화 |
| **leejunggyun** | 30% | 모델 구현 (6개 알고리즘) |
| **Gyu1026** | 20% | 평가 시스템, 성능 분석 |

**Total**: 100%

---

### 🤝 Collaboration Process

1. **Week 1-2**: 프로젝트 기획 및 데이터 수집
   - 전체 회의로 목표 설정
   - 데이터 소스 확정 (Kaggle Steam dataset)
   - 역할 분담

2. **Week 3-4**: 데이터 전처리 및 EDA
   - Lim-K-M이 전처리 파이프라인 구축
   - bk11052가 데이터 병합 및 검증
   - 주간 리뷰 미팅으로 진행 상황 공유

3. **Week 5-6**: 모델 구현
   - leejunggyun이 6개 알고리즘 구현
   - Gyu1026가 평가 시스템 구축
   - 코드 리뷰 및 피드백

4. **Week 7**: 최종 통합 및 발표 준비
   - bk11052가 main 브랜치 통합
   - 전체 팀이 결과 분석 및 PPT 작성
   - 리허설 및 피드백

---

### 💡 Teamwork Lessons Learned

#### 잘된 점
- **명확한 역할 분담**: 각자 전문성 발휘
- **Git 브랜치 활용**: 독립적으로 작업 가능
- **정기적 소통**: 주간 미팅으로 방향성 유지

#### 개선할 점
- **초기 설계 부족**: 프로젝트 구조를 중간에 재설계
- **브랜치 전략 미흡**: 브랜치 병합 시 혼란
- **일정 관리**: 모델 실행 시간 과소평가

---

## 9️⃣ Open-Source SW

### 🏗️ Architecture Description

#### 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    ML Game Recommendation                │
│                         System                           │
└─────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
        ┌───────▼──────┐        ┌──────▼───────┐
        │  Data Layer  │        │ Model Layer  │
        └───────┬──────┘        └──────┬───────┘
                │                       │
                │                       │
┌───────────────▼───────────────┐       │
│  Raw Data Sources              │       │
│  - games.csv (50,872 games)   │       │
│  - output.csv (443,144 reviews)│      │
└───────────────┬───────────────┘       │
                │                       │
                │                       │
┌───────────────▼───────────────┐       │
│  Data Integration              │       │
│  (src/data_integration.py)    │       │
│  - Merge games + reviews       │       │
│  - Result: merged_data.csv     │       │
└───────────────┬───────────────┘       │
                │                       │
                │                       │
┌───────────────▼───────────────┐       │
│  Data Preprocessing            │       │
│  (src/data_preprocessing.py)  │       │
│  - Clean, Transform, Encode    │       │
│  - Result: preprocessed_data.csv│      │
└───────────────┬───────────────┘       │
                │                       │
                │                       │
┌───────────────▼───────────────┐       │
│  Exploratory Data Analysis     │       │
│  (src/data_eda.py)            │       │
│  - 10 visualizations           │       │
│  - Statistical analysis        │       │
└───────────────────────────────┘       │
                                        │
                ┌───────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│              Recommendation Models             │
│                                                │
│  ┌────────────────────────────────────────┐  │
│  │ 1. CF Baseline                         │  │
│  │    (models/cf_baseline.py)             │  │
│  │    - Matrix Factorization              │  │
│  └────────────────────────────────────────┘  │
│                                                │
│  ┌────────────────────────────────────────┐  │
│  │ 2. CBF Baseline                        │  │
│  │    (models/cbf_baseline.py)            │  │
│  │    - TF-IDF + Cosine Similarity        │  │
│  └────────────────────────────────────────┘  │
│                                                │
│  ┌────────────────────────────────────────┐  │
│  │ 3. Hybrid Simple                       │  │
│  │    (models/hybrid_simple.py)           │  │
│  │    - Weighted Combination (0.3 + 0.7)  │  │
│  └────────────────────────────────────────┘  │
│                                                │
│  ┌────────────────────────────────────────┐  │
│  │ 4. Hybrid Stacking ⭐                  │  │
│  │    (models/hybrid_stacking.py)         │  │
│  │    - LogisticRegression Meta-Learner   │  │
│  └────────────────────────────────────────┘  │
│                                                │
│  ┌────────────────────────────────────────┐  │
│  │ 5. Factorization Machines              │  │
│  │    (models/factorization_machines.py)  │  │
│  │    - Feature Interaction Modeling      │  │
│  └────────────────────────────────────────┘  │
│                                                │
│  ┌────────────────────────────────────────┐  │
│  │ 6. Word Embeddings                     │  │
│  │    (models/word_embeddings.py)         │  │
│  │    - Word2Vec Semantic Similarity      │  │
│  └────────────────────────────────────────┘  │
│                                                │
└────────────────────┬───────────────────────────┘
                     │
                     │
┌────────────────────▼───────────────────────────┐
│         Evaluation Framework                   │
│         (src/run_eval_preprocessed.py)         │
│                                                 │
│  - Precision@K, Recall@K, F1@K                │
│  - NDCG@K                                      │
│  - Coverage                                    │
│  - Running Time                                │
│                                                 │
│  → results/model_comparison.csv                │
└────────────────────┬───────────────────────────┘
                     │
                     │
┌────────────────────▼───────────────────────────┐
│              Results & Visualization            │
│                                                 │
│  - results/eda_visualizations/ (10 plots)      │
│  - results/model_comparison.csv                │
│  - results/performance_charts/                 │
└─────────────────────────────────────────────────┘
```

---

### 📦 Directory Structure

```
ML/
├── README.md                    # 프로젝트 문서
├── requirements.txt             # Python 의존성
├── .gitignore                   # Git 제외 파일
│
├── data/                        # 데이터 디렉토리
│   ├── .gitkeep
│   ├── games.csv                # 게임 메타데이터 (ignored)
│   ├── output.csv               # 리뷰 데이터 (ignored)
│   ├── merged_data.csv          # 병합 데이터 (ignored)
│   └── preprocessed_data.csv    # 전처리 데이터 (ignored)
│
├── src/                         # 소스 코드
│   ├── data_integration.py      # 데이터 병합
│   ├── data_preprocessing.py    # 데이터 전처리
│   ├── data_eda.py              # EDA 및 시각화
│   └── run_eval_preprocessed.py # 평가 프레임워크
│
├── models/                      # 추천 모델
│   ├── cf_baseline.py           # Collaborative Filtering
│   ├── cbf_baseline.py          # Content-Based Filtering
│   ├── hybrid_simple.py         # Weighted Hybrid
│   ├── hybrid_stacking.py       # Stacking Hybrid
│   ├── factorization_machines.py # Factorization Machines
│   └── word_embeddings.py       # Word2Vec
│
└── results/                     # 결과 파일
    ├── .gitkeep
    └── eda_visualizations/      # EDA 시각화
        ├── 01_missing_values_heatmap.png
        ├── 02_rating_distribution.png
        ├── 03_positive_ratio_distribution.png
        ├── 04_user_reviews_distribution.png
        ├── 05_price_distribution.png
        ├── 06_platform_distribution.png
        ├── 07_releases_by_year.png
        ├── 08_top_20_games.png
        ├── 09_ratio_vs_reviews.png
        └── 10_sentiment_distribution.png
```

---

### 🔧 Technology Stack

#### Programming Language
- **Python 3.8+**: 메인 언어

#### Data Processing
- **pandas 1.5.0+**: 데이터 조작 및 분석
- **numpy 1.23.0+**: 수치 계산

#### Machine Learning
- **scikit-learn 1.2.0+**: 모델링, 전처리, 평가
- **scipy 1.9.0+**: 희소 행렬, 과학 계산

#### Natural Language Processing
- **gensim 4.3.0+**: Word2Vec 임베딩

#### Visualization
- **matplotlib 3.6.0+**: 기본 시각화
- **seaborn 0.12.0+**: 고급 통계 시각화

#### Utilities
- **tqdm 4.64.0+**: 진행률 표시

#### Version Control
- **Git**: 버전 관리
- **GitHub**: 원격 저장소

---

### 🌐 GitHub Repository

**URL**: [https://github.com/bk11052/ML_Game_Recommendation](https://github.com/bk11052/ML_Game_Recommendation)

#### Repository Structure
- **main** branch: 최종 완성 코드
- **data/integration** branch: 데이터 병합 실험
- **data/preprocessing** branch: 전처리 실험
- **data/eda** branch: EDA 분석
- **evaluation** branch: 평가 시스템
- **Hybrid-with-stacking** branch: Stacking 모델 개발
- **leejunggyun** branch: 모델 실험
- **Lim-K-M** branch: 이전 코드 백업

#### Key Files
1. **README.md**: 프로젝트 개요 및 사용법
2. **requirements.txt**: 패키지 의존성
3. **src/data_*.py**: 데이터 파이프라인
4. **models/*.py**: 추천 알고리즘 구현
5. **results/**: 실험 결과 및 시각화

---

### 📜 License
**Educational Use Only** - 학습 목적 프로젝트

---

## 🎯 Conclusion

### 프로젝트 요약
- **6가지 추천 알고리즘** 구현 및 비교
- **250,423개 데이터** 전처리 및 분석
- **10개 시각화**로 데이터 인사이트 도출
- **체계적인 평가 프레임워크** 구축

### 기대 효과
1. **학술적 기여**: 추천 알고리즘 비교 연구
2. **실무 적용 가능성**: Steam 게임 추천 시스템 구축 가능
3. **학습 경험**: 팀 프로젝트 협업 역량 강화

### Next Steps
1. 모델 실행 및 결과 수집
2. 성능 분석 및 최적 모델 선정
3. 웹 애플리케이션 배포 (향후 계획)

---

**Thank You!**

**Questions?**
