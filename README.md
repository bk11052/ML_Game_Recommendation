# 🎮 ML Game Recommendation System

Steam 게임 추천 시스템: 다양한 알고리즘 비교 및 성능 분석

## 📊 데이터셋

- **50,872개 게임**
- **443,144개 리뷰**
- `data/merged_data.csv`: games.csv와 output.csv를 병합한 데이터셋

## 🎯 프로젝트 목표

협업 필터링(CF), 콘텐츠 기반 필터링(CBF), 하이브리드 등 6가지 추천 알고리즘을 구현하고,
동일한 데이터셋에서 성능을 비교 분석하여 최적의 모델을 도출합니다.

## 📂 프로젝트 구조

```
ML/
├── README.md
├── requirements.txt
├── data/
│   └── merged_data.csv          # 병합된 데이터
├── src/
│   ├── data_integration.py      # 데이터 병합
│   ├── data_preprocessing.py    # 데이터 전처리
│   ├── data_eda.py              # 탐색적 데이터 분석
│   └── run_eval_preprocessed.py # 평가 프레임워크
├── models/
│   ├── cf_baseline.py           # CF 베이스라인
│   ├── cbf_baseline.py          # CBF 베이스라인
│   ├── hybrid_simple.py         # 단순 하이브리드 (가중치 결합)
│   ├── hybrid_stacking.py       # Stacking 하이브리드 (학습 기반)
│   ├── factorization_machines.py # Factorization Machines
│   └── word_embeddings.py       # Word2Vec 기반 CBF
└── results/
    └── .gitkeep                 # 평가 결과 저장 위치
```

## 🔧 설치 및 실행

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# 데이터 병합
python src/data_integration.py

# 데이터 전처리
python src/data_preprocessing.py
```

### 3. 탐색적 데이터 분석
```bash
python src/data_eda.py
```

### 4. 모델 실행
```bash
# CF 베이스라인
python models/cf_baseline.py

# CBF 베이스라인
python models/cbf_baseline.py

# 하이브리드 모델들
python models/hybrid_simple.py
python models/hybrid_stacking.py

# 고급 모델들
python models/factorization_machines.py
python models/word_embeddings.py
```

## 📈 평가 지표

- **Precision@K**: Top-K 추천의 정확도
- **Recall@K**: Top-K 추천의 재현율
- **F1@K**: Precision과 Recall의 조화 평균
- **NDCG@K**: 순위를 고려한 평가 지표
- **Coverage**: 추천 다양성
- **Running Time**: 실행 시간

## 🎓 모델 설명

### 1. CF Baseline (Collaborative Filtering)
- Matrix Factorization with SGD
- 유저-아이템 상호작용 기반 추천

### 2. CBF Baseline (Content-Based Filtering)
- TF-IDF + Cosine Similarity
- 게임 제목 기반 콘텐츠 유사도

### 3. Hybrid Simple
- 가중치 기반 결합: CF 30% + CBF 70%
- 단순하지만 효과적인 하이브리드 접근

### 4. Hybrid Stacking ⭐
- LogisticRegression을 통한 학습 기반 결합
- CF와 CBF의 예측을 메타 피처로 사용
- 최적 가중치 자동 학습

### 5. Factorization Machines
- 피처 조합을 통한 고급 예측
- 게임 메타데이터 활용

### 6. Word Embeddings
- Word2Vec 기반 의미적 유사도
- 게임 제목의 임베딩 벡터 활용

## 🌿 브랜치 구조

- `main`: 최종 완성 코드
- `data/integration`: 데이터 병합 실험
- `data/preprocessing`: 데이터 전처리 실험
- `data/eda`: 탐색적 데이터 분석
- `evaluation`: 평가 프레임워크
- `Hybrid-with-stacking`: Stacking 모델 개발
- `leejunggyun`: 다양한 모델 실험

## 👥 Contributors

- bk11052
- Lim-K-M
- leejunggyun
- Gyu1026

## 📝 License

This project is for educational purposes.
