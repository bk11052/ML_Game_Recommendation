import numpy as np, pandas as pd

# =========================================================
# 1. 기본 Precision / Recall / nDCG / MAP 계산 함수
# =========================================================
def precision_at_k(pred, gt, k):
    if k==0: return 0.0
    pred_k = pred[:k]; return sum(i in gt for i in pred_k)/k

def recall_at_k(pred, gt, k):
    return 0.0 if len(gt)==0 else sum(i in gt for i in pred[:k]) / len(gt)

def dcg_at_k(pred, gt, k):
    s=0.0
    for idx,item in enumerate(pred[:k], start=1):
        if item in gt:
            s += 1.0/np.log2(idx+1)
    return s

def ndcg_at_k(pred, gt, k):
    ih = min(len(gt), k)
    if ih==0: return 0.0
    idcg = sum(1.0/np.log2(i+1) for i in range(1, ih+1))
    return dcg_at_k(pred, gt, k)/idcg

# =========================================================
# 2. MAP@K 계산 (Average Precision)
# =========================================================
def apk(pred, gt, k):
    if len(gt)==0: return 0.0
    score, hit = 0.0, 0
    for idx, item in enumerate(pred[:k], start=1):
        if item in gt:
            hit += 1; score += hit/idx
    return score / min(len(gt), k)

def mapk(preds_dict, gts_dict, k):
    return float(np.mean([apk(preds_dict.get(u,[]), gts_dict[u], k) for u in gts_dict])) if gts_dict else 0.0

# =========================================================
# 3. 종합 지표 DataFrame 생성
# =========================================================
def compute_topk(preds, gts, ks=(5,10,20)):
    rows=[]
    for k in ks:
        prec = np.mean([precision_at_k(preds.get(u,[]), gts[u], k) for u in gts])
        rec  = np.mean([recall_at_k   (preds.get(u,[]), gts[u], k) for u in gts])
        ndcg = np.mean([ndcg_at_k     (preds.get(u,[]), gts[u], k) for u in gts])
        mp   = mapk(preds, gts, k)
        rows.append({"K":k,"Precision":prec,"Recall":rec,"nDCG":ndcg,"MAP":mp})
    return pd.DataFrame(rows)

# =========================================================
# 4. RMSE / MAE (회귀형 보조지표)
# =========================================================
def rmse(y_true,y_pred):
    y_true=np.asarray(y_true,float); y_pred=np.asarray(y_pred,float)
    return float(np.sqrt(np.mean((y_true-y_pred)**2)))

def mae(y_true,y_pred):
    y_true=np.asarray(y_true,float); y_pred=np.asarray(y_pred,float)
    return float(np.mean(np.abs(y_true-y_pred)))
