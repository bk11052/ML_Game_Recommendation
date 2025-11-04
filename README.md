# ML_Game_Recommendation

#Gemini 활용한 코드 설명
1단계: '전문가' 2명 훈련 (Level 0)
추천을 하기 전에, 먼저 2명의 '전문가'가 훈련 데이터(preprocessed_data.csv)를 학습합니다.

CF 전문가 (MatrixFactorization):

모든 유저와 게임 간의 평점(상호작용)을 학습합니다.

전문 분야: 유저의 과거 행동 패턴 (CF)

CBF 전문가 (TF-IDF + Cosine):

모든 게임 제목(콘텐츠)의 유사도를 학습합니다.

전문 분야: 게임 콘텐츠의 유사성 (CBF)

2단계: '매니저' 1명 훈련 (Level 1 - Stacking)
이 부분이 수업에서 배우지 않은 핵심입니다. 0.3, 0.7처럼 가중치를 찍는 대신, '매니저'(LogisticRegression)를 훈련시켜 최적의 조합법을 스스로 학습하게 합니다.

'매니저'용 훈련 교재 만들기:

모든 훈련 데이터(필터링된 리뷰)를 하나씩 살펴보며, CF 전문가와 CBF 전문가에게 물어봅니다.

"유저 A가 게임 X에 '긍정' 평가를 했는데, CF 전문가님 생각은 몇 점인가요?" ➡️ cf_score (예: 0.8)

"CBF 전문가님 생각은요?" ➡️ cbf_score (예: 0.7)

이 예측 점수(cf_score, cbf_score)와 실제 정답(rating)을 한 줄로 묶어 '교재'(X_meta, y_meta)를 만듭니다.

'매니저' 훈련:

'매니저'(LogisticRegression)가 이 교재를 통째로 학습합니다.

학습 목표: cf_score와 cbf_score가 얼마일 때 실제 '긍정'(1)이 나왔는지 그 **패턴(최적의 가중치)**을 학습합니다.

3단계: 실제 추천 과정 (유저가 게임 추천을 요청할 때)
test_user_id (예: '유저 A')에게 게임 5개를 추천하는 과정은 다음과 같습니다.

'전문가' 의견 수렴:

'유저 A'가 아직 안 해본 **모든 게임(49개)**에 대해 두 전문가가 각각 점수를 매깁니다.

CF 전문가: "유저 A의 패턴을 보니 49개 게임의 CF 점수는 [0.8, 0.2, ..., 0.9]입니다."

CBF 전문가: "유저 A가 좋아한 게임과 비교하니 CBF 점수는 [0.7, 0.3, ..., 0.8]입니다."

'매니저'에게 보고:

이 두 개의 점수 목록(49개 게임 x 2개 점수)을 '매니저'에게 전달합니다.

'매니저'의 최종 결정:

'매니저'는 이미 훈련(2단계)을 통해 최적의 가중치를 알고 있습니다.

LogisticRegression 모델이 49개 게임 각각에 대해 CF 점수와 CBF 점수를 조합하여 **"이 유저가 이 게임을 '긍정' 평가할 최종 확률"**을 0.0에서 1.0 사이로 계산합니다.

최종 추천:

이 최종 확률(예측 확률)이 1.0에 가장 가까운 순서대로 게임을 정렬합니다.

Top 5 게임을 뽑아 유저에게 보여줍니다.

# 데이터가 너무 많아 시간이 오래 걸릴 시 최소 리뷰수 2로 수
print(f"    -> 원본 상호작용(리뷰) 수: {len(ratings_df)}")
user_counts = ratings_df['author_id'].value_counts()
active_users = user_counts[user_counts >= 2].index
ratings_df_final = ratings_df[ratings_df['author_id'].isin(active_users)].copy()

