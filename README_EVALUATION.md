
# ⚙️ Evaluation – Hybrid Game Recommendation System

게임 추천 시스템의 최종 성능을 검증하기 위한 **평가 단계(Evaluation)** 설명입니다.

## 📊 평가 개요

본 단계에서는 전처리된 데이터(`preprocessed_data.csv`)를 이용하여  
**Hybrid Recommendation System (CBF + CF)** 의 성능을 정량적으로 측정하였습니다.

### ✅ 평가 목적

- **추천 정확도(Precision)** 및 **재현율(Recall)** 을 기반으로 모델 성능 평가  
- **하이브리드 가중치(α=0.7)** 적용에 따른 추천 품질 검증  
- **탐색형 추천(Exploratory Recommendation)** 특성 분석  

---

## 📁 데이터셋

- **입력:** `preprocessed_data.csv`  
  (전처리 단계에서 생성된 통합 데이터셋)  
- **주요 컬럼:**  
  `author_id`, `app_id`, `title`, `is_positive_encoded`, `rating_*`, `price_final`, `positive_ratio`, `discount` 등  
- **Holdout 전략:** 각 사용자별로 하나의 테스트 아이템을 분리하여 검증 (`HOLDOUT=1`)

---

## 🧠 평가 스크립트

### 1️⃣ 실행 명령어
```bash
python run_evaluation.py
```

### 2️⃣ 주요 기능
- 사용자별 훈련/테스트 데이터 분리  
- 하이브리드 점수 계산  
  ```
  Hybrid = α * Content-Based + (1 - α) * Popularity
  ```
- 각 사용자에 대해 추천 리스트 생성  
- Top-K(5, 10, 20) 단위로 Precision@K, Recall@K, F1-score 계산  
- 결과 저장:  
  - `eval_outputs/per_user_metrics.csv`  
  - `eval_outputs/summary_metrics.csv`  
  - `eval_outputs/split_info.json`

---

## 📈 결과 요약

| Metric | @5 | @10 | @20 |
|---------|----|-----|-----|
| **Precision** | 0.039 | 0.033 | 0.035 |
| **Recall** | 0.193 | 0.333 | 0.699 |
| **F1-score** | 0.064 | 0.061 | 0.067 |

### 🔍 해석
- **Precision:** 낮지만 일정함 → 추천의 다양성 확보  
- **Recall:** K가 커질수록 상승 → 더 많은 실제 선호 아이템 포함  
- **F1-score:** 두 지표의 균형, K=20에서 최고치  
- 즉, **정확도보다 다양성을 중시하는 탐색형 추천 시스템** 구조를 보임  

---

## 🧭 시각화 결과

- 📊 **K별 Precision / Recall / F1 변화 추세**
- 🔄 **Precision–Recall Trade-off 곡선**

> 그래프를 통해 추천 리스트의 길이에 따른 성능 변화를 한눈에 확인할 수 있으며,  
> 하이브리드 모델의 탐색적 추천 경향(Recall 우세)을 명확히 확인함.

---

## 💬 결론

> 본 하이브리드 추천 시스템은 정확도(Precision)는 낮지만,  
> 다양한 게임을 포함해 사용자의 잠재적 관심 영역을 넓히는  
> **탐색 중심(Exploratory) 추천 시스템**으로 작동함을 확인하였다.  
> F1-score는 안정적으로 유지되어 추천 품질의 일관성을 보였으며,  
> Cold-start 및 희소 데이터 환경에서도 유효한 추천이 가능함을 검증하였다.

---

## 🚀 향후 개선 방향

| 개선 방향 | 설명 | 기대 효과 |
|------------|------|------------|
| **1. Word Embedding 강화** | TF-IDF → Word2Vec/BERT 적용 | 의미 기반 추천 향상 |
| **2. α 가중치 동적 조정** | 사용자 피드백 기반 학습 | 개인화 추천 강화 |
| **3. Reinforcement Learning** | 보상 기반 추천 반복 학습 | Active Learning 요소 반영 |
| **4. Precision 향상** | CBF에 감성 분석 추가 | 정확도 개선 및 개인화 심화 |

---

## 👥 Contributors

- 김병규 (Team Leader)  
- 김인규 (Evaluation & Visualization)  
- 이정균 (Modeling)  
- 임규민 (Data Integration & Preprocessing)
