
# cf_mf_script_full.py
"""
完整实现：
- User-based CF (cosine + top-k)
- Item-based CF (cosine + top-k)
- FunkSVD (plain MF via SGD)
- Bias-SVD (with user/item biases via SGD)
会读取 steam-200k.csv，训练并评估（MAE, RMSE），并保存结果到 cf_mf_results.csv
"""

import os
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array

import pandas as pd
from pathlib import Path

# Path to the CSV file: steam-200k.csv/steam-200k.csv
csv_path = 'steam-200k.csv'
print('Loading:', csv_path)
# low_memory=False avoids dtype inference warnings for large files
df = pd.read_csv(csv_path, low_memory=False)

print('Columns:')
for col in df.columns:
    print(col)
# Rename first four columns and delete the last column
# Assume df is already loaded
cols = list(df.columns)
print('Original first four columns:', cols[:4])
# Create rename mapping
rename_map = {
    cols[0]: 'user-id',
    cols[1]: 'game-title',
    cols[2]: 'behavior-name',
    cols[3]: 'value'
}
# Rename and drop last column
df = df.rename(columns=rename_map)
df = df.drop(columns=[cols[-1]])

print('\nModified column names (line by line):')
for c in df.columns:
    print(c)

# Display first 5 rows for verification
df.head()
# ---------- 配置（可按需修改） ----------
# DATA_PATH = "../steam-200k.csv"
RESULTS_PATH = "cf_mf_results.csv"

# CF 超参数
CF_TOPK = 20

# MF 超参数（FunkSVD）
MF_LATENT = 20
MF_LR = 0.001  # 降低学习率防止溢出
MF_REG = 0.05   # 增加正则化
MF_EPOCHS = 10

# Bias-SVD 超参数
BIAS_LATENT = 20
BIAS_LR = 0.0005  # 降低学习率
BIAS_REG = 0.05    # 增加正则化
BIAS_EPOCHS = 15

# 评分缩放配置
SCALE_RATINGS = True  # 缩放评分到[0,1]或[-1,1]范围
SCALE_METHOD = 'minmax'  # 'minmax' 或 'standard'

RANDOM_STATE = 42
# ----------------------------------------

# 设置随机种子
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ---------- 辅助函数 ----------
def remove_nan_inf(arr):
    """移除数组中的NaN和Inf值"""
    arr = np.array(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def scale_ratings(ratings, method='minmax'):
    """缩放评分以防止数值溢出"""
    ratings = np.array(ratings, dtype=np.float64).reshape(-1, 1)
    
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0.1, 1.0))  # 避免0值
    else:
        scaler = StandardScaler()
    
    scaled_ratings = scaler.fit_transform(ratings).flatten()
    return scaled_ratings, scaler

def inverse_scale_ratings(scaled_ratings, scaler):
    """将缩放后的评分转换回原始范围"""
    if scaler is None:
        return scaled_ratings
    
    scaled_ratings = np.array(scaled_ratings, dtype=np.float64).reshape(-1, 1)
    original_ratings = scaler.inverse_transform(scaled_ratings).flatten()
    return original_ratings

# ---------- 余弦相似度函数（修复NaN问题） ----------
def cosine_sim_matrix(mat, axis=0):
    """
    mat: 2D numpy array (rows: users, cols: items)
    axis=0: compute similarity between rows (users)
    axis=1: compute similarity between cols (items) -- implemented by transposing inside
    返回：相似度矩阵 (n x n)
    """
    if axis == 1:
        mat = mat.T
    
    # 确保矩阵中没有NaN或Inf
    mat = remove_nan_inf(mat)
    
    # 计算向量范数
    norms = np.linalg.norm(mat, axis=1)
    # 防止除零
    norms[norms == 0] = 1e-9
    
    # 计算点积
    dot_product = mat @ mat.T
    
    # 计算相似度矩阵
    sim = dot_product / (norms[:, None] * norms[None, :])
    
    # 处理可能的NaN/Inf
    sim = remove_nan_inf(sim)
    
    np.fill_diagonal(sim, 0.0)
    return sim

# ---------- User-based CF: 逐条预测（修复NaN问题） ----------
def predict_user_cf_single(user_id, item_id, sim_u, train_mat, topk=20, global_mean=0.0):
    # 确保没有NaN
    global_mean = float(global_mean) if not np.isnan(global_mean) else 0.0
    
    # 找出对该 item 有评分的用户
    users_who_rated = np.where(train_mat[:, item_id] > 0)[0]
    if users_who_rated.size == 0:
        return global_mean
    
    # 获取相似度并过滤NaN
    sims = sim_u[user_id, users_who_rated]
    sims = remove_nan_inf(sims)
    
    if np.sum(np.abs(sims)) == 0:
        return global_mean
    
    # 取 top-k 相似用户（按 sim 值）
    k = min(topk, sims.size)
    topk_idx = np.argsort(sims)[-k:]
    
    numer = 0.0
    denom = 0.0
    
    for idx in topk_idx:
        u = users_who_rated[idx]
        s = sims[idx]
        r = train_mat[u, item_id]
        
        # 确保值有效
        if np.isnan(r) or np.isinf(r):
            continue
            
        numer += s * r
        denom += abs(s)
    
    if denom == 0:
        return global_mean
    
    pred = numer / denom
    
    # 确保预测值有效
    if np.isnan(pred) or np.isinf(pred):
        return global_mean
    
    return pred

def batch_predict_user_cf(test_df, sim_u, train_mat, k=20, global_mean=0.0):
    preds = []
    global_mean = float(global_mean) if not np.isnan(global_mean) else 0.0
    
    for _, row in test_df.iterrows():
        try:
            p = predict_user_cf_single(int(row.user_id), int(row.item_id), sim_u, train_mat, topk=k, global_mean=global_mean)
        except:
            p = global_mean
        preds.append(p)
    
    return np.array(preds)

# ---------- Item-based CF（修复NaN问题） ----------
def predict_item_cf_single(user_id, item_id, sim_i, train_mat, topk=20, global_mean=0.0):
    # 确保没有NaN
    global_mean = float(global_mean) if not np.isnan(global_mean) else 0.0
    
    # 找出用户评分过的物品
    items_rated = np.where(train_mat[user_id, :] > 0)[0]
    if items_rated.size == 0:
        return global_mean
    
    # 获取相似度并过滤NaN
    sims = sim_i[item_id, items_rated]
    sims = remove_nan_inf(sims)
    
    if np.sum(np.abs(sims)) == 0:
        return global_mean
    
    # 取 top-k 相似物品
    k = min(topk, sims.size)
    topk_idx = np.argsort(sims)[-k:]
    
    numer = 0.0
    denom = 0.0
    
    for idx in topk_idx:
        j = items_rated[idx]
        s = sims[idx]
        r = train_mat[user_id, j]
        
        if np.isnan(r) or np.isinf(r):
            continue
            
        numer += s * r
        denom += abs(s)
    
    if denom == 0:
        return global_mean
    
    pred = numer / denom
    
    if np.isnan(pred) or np.isinf(pred):
        return global_mean
    
    return pred

def batch_predict_item_cf(test_df, sim_i, train_mat, k=20, global_mean=0.0):
    preds = []
    global_mean = float(global_mean) if not np.isnan(global_mean) else 0.0
    
    for _, row in test_df.iterrows():
        try:
            p = predict_item_cf_single(int(row.user_id), int(row.item_id), sim_i, train_mat, topk=k, global_mean=global_mean)
        except:
            p = global_mean
        preds.append(p)
    
    return np.array(preds)

# ---------- FunkSVD (plain MF via SGD) ----------
def train_funksvd(train_df, n_users, n_items, n_factors=20, lr=0.01, reg=0.02, n_epochs=10, verbose=False):
    # 初始化因子矩阵（更小的初始值）
    P = 0.01 * np.random.randn(n_users, n_factors)
    Q = 0.01 * np.random.randn(n_items, n_factors)
    
    # 确保没有NaN
    P = remove_nan_inf(P)
    Q = remove_nan_inf(Q)
    
    # 只使用有效的评分数据
    train_data = train_df.dropna(subset=['user_id', 'item_id', 'rating'])
    
    for epoch in range(n_epochs):
        # 随机打乱样本
        for _, row in train_data.sample(frac=1.0, random_state=None).iterrows():
            try:
                u = int(row.user_id)
                i = int(row.item_id)
                r = float(row.rating)
                
                # 跳过无效值
                if np.isnan(r) or np.isinf(r) or r <= 0:
                    continue
                
                pred = P[u].dot(Q[i])
                e = r - pred
                
                # 更新因子（添加梯度裁剪）
                grad_P = e * Q[i] - reg * P[u]
                grad_Q = e * P[u] - reg * Q[i]
                
                # 梯度裁剪防止爆炸
                grad_P = np.clip(grad_P, -1.0, 1.0)
                grad_Q = np.clip(grad_Q, -1.0, 1.0)
                
                P[u] += lr * grad_P
                Q[i] += lr * grad_Q
                
                # 确保值有效
                P[u] = remove_nan_inf(P[u])
                Q[i] = remove_nan_inf(Q[i])
                
            except:
                continue
        
        if verbose:
            try:
                preds = []
                for _, r in train_data.iterrows():
                    try:
                        pred = P[int(r.user_id)].dot(Q[int(r.item_id)])
                        preds.append(pred)
                    except:
                        preds.append(0)
                
                preds = remove_nan_inf(preds)
                valid_ratings = remove_nan_inf(train_data['rating'].values)
                
                rmse = math.sqrt(mean_squared_error(valid_ratings, preds))
                print(f"[FunkSVD] Epoch {epoch+1}/{n_epochs} train RMSE: {rmse:.4f}")
            except:
                print(f"[FunkSVD] Epoch {epoch+1}/{n_epochs} completed")
    
    return P, Q

def predict_funksvd(P, Q, test_df, global_mean=None):
    preds = []
    global_mean = float(global_mean) if global_mean is not None and not np.isnan(global_mean) else 0.0
    
    for _, row in test_df.iterrows():
        try:
            u = int(row.user_id)
            i = int(row.item_id)
            
            if u >= len(P) or i >= len(Q):
                p = global_mean
            else:
                p = P[u].dot(Q[i])
                
                # 确保预测值有效
                if np.isnan(p) or np.isinf(p):
                    p = global_mean
                    
        except:
            p = global_mean
        
        preds.append(p)
    
    return np.array(preds)

# ---------- Bias-SVD ----------
def train_bias_svd(train_df, n_users, n_items, n_factors=20, lr=0.005, reg=0.02, n_epochs=15, verbose=False):
    # 只使用有效的评分数据
    train_data = train_df.dropna(subset=['user_id', 'item_id', 'rating'])
    valid_ratings = remove_nan_inf(train_data['rating'].values)
    
    mu = np.mean(valid_ratings) if len(valid_ratings) > 0 else 0.0
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    
    # 初始化因子矩阵（更小的初始值）
    P = 0.01 * np.random.randn(n_users, n_factors)
    Q = 0.01 * np.random.randn(n_items, n_factors)
    
    P = remove_nan_inf(P)
    Q = remove_nan_inf(Q)
    
    for epoch in range(n_epochs):
        # 随机打乱样本
        for _, row in train_data.sample(frac=1.0, random_state=None).iterrows():
            try:
                u = int(row.user_id)
                i = int(row.item_id)
                r = float(row.rating)
                
                if np.isnan(r) or np.isinf(r) or r <= 0:
                    continue
                
                pred = mu + bu[u] + bi[i] + P[u].dot(Q[i])
                e = r - pred
                
                # 更新偏置和因子（添加梯度裁剪）
                bu[u] += lr * np.clip(e - reg * bu[u], -1.0, 1.0)
                bi[i] += lr * np.clip(e - reg * bi[i], -1.0, 1.0)
                
                grad_P = e * Q[i] - reg * P[u]
                grad_Q = e * P[u] - reg * Q[i]
                
                grad_P = np.clip(grad_P, -1.0, 1.0)
                grad_Q = np.clip(grad_Q, -1.0, 1.0)
                
                P[u] += lr * grad_P
                Q[i] += lr * grad_Q
                
                # 确保值有效
                bu[u] = 0 if np.isnan(bu[u]) or np.isinf(bu[u]) else bu[u]
                bi[i] = 0 if np.isnan(bi[i]) or np.isinf(bi[i]) else bi[i]
                P[u] = remove_nan_inf(P[u])
                Q[i] = remove_nan_inf(Q[i])
                
            except:
                continue
        
        if verbose:
            try:
                preds = []
                for _, r in train_data.iterrows():
                    try:
                        u = int(r.user_id)
                        i = int(r.item_id)
                        pred = mu + bu[u] + bi[i] + P[u].dot(Q[i])
                        preds.append(pred)
                    except:
                        preds.append(mu)
                
                preds = remove_nan_inf(preds)
                valid_ratings = remove_nan_inf(train_data['rating'].values)
                
                rmse = math.sqrt(mean_squared_error(valid_ratings, preds))
                print(f"[BiasSVD] Epoch {epoch+1}/{n_epochs} train RMSE: {rmse:.4f}")
            except:
                print(f"[BiasSVD] Epoch {epoch+1}/{n_epochs} completed")
    
    return mu, bu, bi, P, Q

def predict_bias_svd(mu, bu, bi, P, Q, test_df):
    preds = []
    mu = float(mu) if not np.isnan(mu) else 0.0
    
    for _, row in test_df.iterrows():
        try:
            u = int(row.user_id)
            i = int(row.item_id)
            
            if u >= len(bu) or i >= len(bi) or u >= len(P) or i >= len(Q):
                p = mu
            else:
                p = mu + bu[u] + bi[i] + P[u].dot(Q[i])
                
                if np.isnan(p) or np.isinf(p):
                    p = mu
                    
        except:
            p = mu
        
        preds.append(p)
    
    return np.array(preds)

# ---------- 主流程（增强数据清洗） ----------

print("数据前5行:")
print(df.head())

# 处理steam-200k数据：只保留play行为，value作为评分（游戏时长）
rating_scaler = None
if 'behavior' in df.columns:
    # 过滤出play行为并移除NaN
    df = df[df['behavior'] == 'play'].copy()
    df = df.dropna(subset=['user_id', 'game_name', 'value'])
    
    # 重命名列以便统一处理
    df = df.rename(columns={'user_id': 'user', 'game_name': 'item', 'value': 'rating'})
    user_col = 'user'
    item_col = 'item'
    rating_col = 'rating'
    print("已过滤出play行为数据，共", len(df), "条记录")
else:
    # 自动选择列
    from sklearn.utils import column_or_1d
    
    def pick_column(df, type_):
        names = {
            "user": ["user", "userid", "user_id", "player", "uid", "account", "username", "steamid"],
            "item": ["item", "itemid", "item_id", "game", "product", "appid", "title", "name"],
            "rating": ["rating", "score", "rate", "stars", "review", "value", "hours", "playtime"]
        }
        for c in df.columns:
            low = c.lower()
            for key in names[type_]:
                if key in low:
                    return c
        # fallback: choose by position
        if type_ == "user":
            return df.columns[0]
        if type_ == "item":
            return df.columns[1] if len(df.columns) > 1 else df.columns[0]
        return df.columns[2] if len(df.columns) > 2 else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    user_col = pick_column(df, "user")
    item_col = pick_column(df, "item")
    rating_col = pick_column(df, "rating")
    print("Detected columns -> user: %s, item: %s, rating: %s" % (user_col, item_col, rating_col))

# 数据清洗：移除所有包含NaN的行
data = df[[user_col, item_col, rating_col]].dropna(how='any')
data = data.rename(columns={user_col: "user", item_col: "item", rating_col: "rating"})

# 转换rating为数值型并过滤无效值
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data = data.dropna(subset=['rating'])
data = data[data['rating'] > 0]  # 只保留正值评分

# 过滤无效的user和item
data = data[data['user'].notna() & data['item'].notna()]
data = data.reset_index(drop=True)

print(f"有效数据条数: {len(data)}")
if len(data) == 0:
    raise ValueError("清洗后没有有效数据")

print(f"原始评分范围: {data['rating'].min():.2f} - {data['rating'].max():.2f}")

# 缩放评分防止数值溢出
if SCALE_RATINGS:
    print(f"\n缩放评分使用 {SCALE_METHOD} 方法...")
    data['rating_scaled'] = scale_ratings(data['rating'].values, SCALE_METHOD)[0]
    rating_scaler = scale_ratings(data['rating'].values, SCALE_METHOD)[1]
    data['rating'] = data['rating_scaled']
    print(f"缩放后评分范围: {data['rating'].min():.4f} - {data['rating'].max():.4f}")

# 编码 user/item 为连续 id
user_enc = LabelEncoder().fit(data['user'])
item_enc = LabelEncoder().fit(data['item'])

data['user_id'] = user_enc.transform(data['user'])
data['item_id'] = item_enc.transform(data['item'])

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)
print(f"n_users={n_users}, n_items={n_items}, n_ratings={len(data)}")

# 划分 train/test
if n_users > 1 and len(data) > 10:
    train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
else:
    train = data.copy()
    test = pd.DataFrame(columns=data.columns)

# 构建 train 矩阵 (numpy)，确保没有NaN
train_mat = np.zeros((n_users, n_items), dtype=np.float64)
for _, row in train.iterrows():
    try:
        u = int(row.user_id)
        i = int(row.item_id)
        r = float(row.rating)
        if r > 0 and not np.isnan(r) and not np.isinf(r):
            train_mat[u, i] = r
    except:
        continue

train_mean = np.mean(train_mat[train_mat > 0]) if np.sum(train_mat > 0) > 0 else 0.0

# 1) 计算相似度
print("\nComputing user-user cosine similarity...")
sim_u = cosine_sim_matrix(train_mat, axis=0)
print("Computing item-item cosine similarity...")
sim_i = cosine_sim_matrix(train_mat, axis=1)

# 2) 训练 MF 模型（FunkSVD）
print("\nTraining FunkSVD (plain MF)...")
P, Q = train_funksvd(train, n_users, n_items, n_factors=MF_LATENT, lr=MF_LR, reg=MF_REG, n_epochs=MF_EPOCHS, verbose=True)

# 3) 训练 Bias-SVD
print("\nTraining Bias-SVD...")
mu, bu, bi, P_b, Q_b = train_bias_svd(train, n_users, n_items, n_factors=BIAS_LATENT, lr=BIAS_LR, reg=BIAS_REG, n_epochs=BIAS_EPOCHS, verbose=True)

# 4) 预测并评估（对 test）
if len(test) == 0:
    print("Warning: test set is empty. 无法评估。")
else:
    print("\nPredicting on test set...")

    # 确保测试集有效
    test = test.dropna(subset=['user_id', 'item_id', 'rating'])
    test['rating'] = pd.to_numeric(test['rating'], errors='coerce')
    test = test.dropna(subset=['rating'])

    preds_usercf = batch_predict_user_cf(test, sim_u, train_mat, k=CF_TOPK, global_mean=train_mean)
    preds_itemcf = batch_predict_item_cf(test, sim_i, train_mat, k=CF_TOPK, global_mean=train_mean)
    preds_funk = predict_funksvd(P, Q, test, global_mean=train_mean)
    preds_bias = predict_bias_svd(mu, bu, bi, P_b, Q_b, test)

    # 处理可能的 NaN/Inf
    def fix_preds(preds):
        arr = np.array(preds, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=train_mean, posinf=train_mean, neginf=train_mean)
        return arr

    preds_usercf = fix_preds(preds_usercf)
    preds_itemcf = fix_preds(preds_itemcf)
    preds_funk = fix_preds(preds_funk)
    preds_bias = fix_preds(preds_bias)

    # 确保真实评分有效
    true_ratings = remove_nan_inf(test['rating'].values)
    
    # 如果评分被缩放，转换回原始范围进行评估
    if SCALE_RATINGS and rating_scaler is not None:
        print("\n转换预测值回原始评分范围...")
        true_ratings = inverse_scale_ratings(true_ratings, rating_scaler)
        preds_usercf = inverse_scale_ratings(preds_usercf, rating_scaler)
        preds_itemcf = inverse_scale_ratings(preds_itemcf, rating_scaler)
        preds_funk = inverse_scale_ratings(preds_funk, rating_scaler)
        preds_bias = inverse_scale_ratings(preds_bias, rating_scaler)

    # 计算评估指标
    mae_usercf = mean_absolute_error(true_ratings, preds_usercf)
    rmse_usercf = math.sqrt(mean_squared_error(true_ratings, preds_usercf))
    mae_itemcf = mean_absolute_error(true_ratings, preds_itemcf)
    rmse_itemcf = math.sqrt(mean_squared_error(true_ratings, preds_itemcf))
    mae_funk = mean_absolute_error(true_ratings, preds_funk)
    rmse_funk = math.sqrt(mean_squared_error(true_ratings, preds_funk))
    mae_bias = mean_absolute_error(true_ratings, preds_bias)
    rmse_bias = math.sqrt(mean_squared_error(true_ratings, preds_bias))

    # 保存结果表
    results = pd.DataFrame([
        ["User-CF (cosine, top-k=%d)" % CF_TOPK, mae_usercf, rmse_usercf],
        ["Item-CF (cosine, top-k=%d)" % CF_TOPK, mae_itemcf, rmse_itemcf],
        ["FunkSVD (latent=%d)" % MF_LATENT, mae_funk, rmse_funk],
        ["Bias-SVD (latent=%d)" % BIAS_LATENT, mae_bias, rmse_bias]
    ], columns=["Model", "MAE", "RMSE"])

    print("\nEvaluation results (原始评分范围):")
    print(results.to_string(index=False))

    try:
        results.to_csv(RESULTS_PATH, index=False)
        print(f"\nSaved results to {RESULTS_PATH}")
    except Exception as e:
        print("保存结果到文件时出错:", e)

    # 可选：将部分预测写回 test（便于查看）
    try:
        test_out = test.copy().reset_index(drop=True)
        test_out['pred_usercf'] = preds_usercf[:len(test_out)]
        test_out['pred_itemcf'] = preds_itemcf[:len(test_out)]
        test_out['pred_funksvd'] = preds_funk[:len(test_out)]
        test_out['pred_biassvd'] = preds_bias[:len(test_out)]
        
        # 添加原始评分（如果有）
        if 'rating_scaled' in test_out.columns:
            test_out['original_rating'] = inverse_scale_ratings(test_out['rating'], rating_scaler)
        
        csv_preds = "cf_mf_test_predictions.csv"
        test_out.to_csv(csv_preds, index=False)
        print("Saved detailed test predictions to", csv_preds)
    except Exception as e:
        print("保存预测文件出错:", e)