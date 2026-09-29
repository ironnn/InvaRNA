import optuna
import lightgbm as lgbm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import joblib
import json
import glob

# ================= 1. 基础配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 111
SAVE_DIR = os.path.join(SCRIPT_DIR, "data/output")
# 专门存放索引文件的子目录
INDICES_DIR = os.path.join(SAVE_DIR, "indices")
os.makedirs(INDICES_DIR, exist_ok=True)

# 路径
PATH_HUMAN = os.path.join(SCRIPT_DIR, "data/input/train_ready_human_regional_strict_20260117_pred_8gpu_bf16.pkl")
PATH_MOUSE = os.path.join(SCRIPT_DIR, "data/input/train_ready_mouse_regional_strict_20260117_pred_8gpu_bf16.pkl")
PATH_MOUSE_TIME = os.path.join(SCRIPT_DIR, "data/input/mouse_time.csv")
PATH_HUMAN_TIME = os.path.join(SCRIPT_DIR, "data/input/human_time.csv")

# ================= 2. 数据准备与索引冻结 =================
def preprocess_human_data(path, path_time):
    print(f"Loading Human data from {path}...")
    df = pd.read_pickle(path)
    df_time = pd.read_csv(path_time)
    df['hl'] = df['pred_score']
    if 'humanTE' in df.columns: df.rename(columns={'humanTE': 'mean_te'}, inplace=True)
    clean_gene_ids = df['human_gene_id'].astype(str).str.split('.').str[0]
    mapping_series = df_time.drop_duplicates(subset=['Ensembl Gene Id']).set_index('Ensembl Gene Id')['half-life (PC1)']
    df['hl'] = clean_gene_ids.map(mapping_series).fillna(df['hl'])
    # ⚠️ 关键：重置索引，确保是 0..N-1 的连续整数，方便后续记录 Index
    return df.reset_index(drop=True)

def preprocess_mouse_data(path_data, path_time):
    print(f"Loading Mouse data from {path_data}...")
    df = pd.read_pickle(path_data)
    df_time = pd.read_csv(path_time)
    df['hl'] = df['pred_score']
    if 'mouseTE' in df.columns: df.rename(columns={'mouseTE': 'mean_te'}, inplace=True)
    df.rename(columns={'gene_id_x': 'gene_id'}, inplace=True)
    clean_gene_ids = df['gene_id'].astype(str).str.split('.').str[0]
    mapping_series = df_time.drop_duplicates(subset=['Ensembl Gene Id']).set_index('Ensembl Gene Id')['half-life (PC1)']
    df['hl'] = clean_gene_ids.map(mapping_series).fillna(df['hl'])
    # ⚠️ 关键：重置索引
    return df.reset_index(drop=True)

def get_sorted_feature_cols(all_columns, ignore_list):
    feat_cols = [c for c in all_columns if c not in ignore_list]
    emb_cols = [c for c in feat_cols if c.startswith('feat_')]
    hl_cols_found = [c for c in feat_cols if 'halflife' in c.lower() or 'hl' in c.lower()]
    hand_cols = [c for c in feat_cols if (c not in emb_cols) and (c not in hl_cols_found)]
    final_order = emb_cols + hand_cols + hl_cols_found
    print(f"🔒 Feature Order Locked: {len(final_order)}")
    return final_order

print(f"🚀 Script started. Saving to: {SAVE_DIR}")

# 加载数据
df_human = preprocess_human_data(PATH_HUMAN, PATH_HUMAN_TIME)
df_mouse = preprocess_mouse_data(PATH_MOUSE, PATH_MOUSE_TIME)
df_human['species'] = 'human'
df_mouse['species'] = 'mouse'

# ================= 3. 数据切分并保存索引 (Index Freezing) =================
print("\n✂️ Splitting datasets...")

# 这里使用了 pandas 的切分，它会保留原始索引（我们刚刚 reset 过的 0..N）
train_df_human, temp_df = train_test_split(df_human, test_size=0.2, random_state=SEED, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, shuffle=True)
train_df_mouse = df_mouse.copy()

# 🧊 保存切分索引
print("🧊 Freezing Indices to disk...")
indices_dict = {
    "human_train_idx": train_df_human.index.tolist(),
    "human_val_idx": val_df.index.tolist(),
    "human_test_idx": test_df.index.tolist()
}
indices_save_path = os.path.join(INDICES_DIR, "split_indices.json")
with open(indices_save_path, 'w') as f:
    json.dump(indices_dict, f)
print(f"✅ Indices saved to: {indices_save_path}")

# 混合数据用于训练
common_cols = [c for c in train_df_human.columns if c in train_df_mouse.columns]
train_df_mixed = pd.concat([train_df_human[common_cols], train_df_mouse[common_cols]], axis=0).reset_index(drop=True)

# 锁定特征
IGNORE_COLS = ['transcript_id', 'gene_id', 'mean_te', 'mouseTE', 'te', 
               'mrna', 'utr5_size', 'cds_size', 'utr3_size', 'mouse_half_life', 
               'pred_score', 'species'] 

sota_feature_order = get_sorted_feature_cols(train_df_mixed.columns, IGNORE_COLS)

# 保存特征列表 (复现必须)
feature_save_path = os.path.join(INDICES_DIR, "feature_names.json")
with open(feature_save_path, 'w') as f:
    json.dump(sota_feature_order, f)
print(f"✅ Feature list saved to: {feature_save_path}")

# 准备矩阵
X_train = train_df_mixed[sota_feature_order]
y_train = train_df_mixed['mean_te']
X_val = val_df[sota_feature_order]
y_val = val_df['mean_te']
X_test = test_df[sota_feature_order]
y_test = test_df['mean_te']

# ================= 4. 定义 "跑完即保存" 的 Objective =================

def objective(trial):
    param = {
        'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
        'n_jobs': 64, 'random_state': SEED,
        'n_estimators': 25000, 
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.025),
        'num_leaves': trial.suggest_int('num_leaves', 60, 110),
        'max_depth': trial.suggest_int('max_depth', 8, 14),
        'min_child_samples': trial.suggest_int('min_child_samples', 40, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'subsample_freq': 1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
    }

    model = lgbm.LGBMRegressor(**param)
    callbacks = [lgbm.early_stopping(stopping_rounds=1500, verbose=False)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=callbacks
    )
    
    best_iter = model.best_iteration_
    val_preds = model.predict(X_val, num_iteration=best_iter)
    test_preds = model.predict(X_test, num_iteration=best_iter)
    
    val_r2 = r2_score(y_val, val_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"Trial {trial.number} | Val: {val_r2:.5f} | Iter: {best_iter}")
    
    # 💾 Snapshot 保存模型
    model_name = f"trial_{trial.number}_TestR2_{test_r2:.5f}.txt"
    pkl_name = f"trial_{trial.number}_TestR2_{test_r2:.5f}.pkl"
    
    model.booster_.save_model(os.path.join(SAVE_DIR, model_name), num_iteration=best_iter)
    joblib.dump(model, os.path.join(SAVE_DIR, pkl_name))
    
    # 保存参数
    with open(os.path.join(SAVE_DIR, f"trial_{trial.number}_params.json"), 'w') as f:
        json.dump(param, f, indent=4)

    return val_r2

# ================= 5. 注入参数并运行 =================

print("\n💉 Preparing Known Candidates...")
known_candidates = [
    # 1. 原始 SOTA
    {'learning_rate': 0.01, 'num_leaves': 66, 'max_depth': 9, 'min_child_samples': 92, 'subsample': 0.8735, 'colsample_bytree': 0.6022, 'reg_alpha': 1.6909, 'reg_lambda': 2.8741},
    # 2. Trial 30 (0.7303)
    {'learning_rate': 0.009739736500790191, 'num_leaves': 61, 'max_depth': 10, 'min_child_samples': 91, 'subsample': 0.7749148157782488, 'colsample_bytree': 0.5891108246163971, 'reg_alpha': 1.4339287547146289, 'reg_lambda': 1.231808497941},
    # 3. 经验参数 C
    {'learning_rate': 0.013021789810756048, 'num_leaves': 63, 'max_depth': 10, 'min_child_samples': 89, 'subsample': 0.7670546800664664, 'colsample_bytree': 0.6300221501639626, 'reg_alpha': 1.437708460113417, 'reg_lambda': 1.1619642294926453},
    # 4. 经验参数 D
    {'learning_rate': 0.016332112274945354, 'num_leaves': 67, 'max_depth': 9, 'min_child_samples': 88, 'subsample': 0.7396776354829975, 'colsample_bytree': 0.6290474302580475, 'reg_alpha': 2.273321802260617, 'reg_lambda': 0.9953947901720361},
    # 5. 经验参数 E
    {'learning_rate': 0.013123632536390602, 'num_leaves': 61, 'max_depth': 10, 'min_child_samples': 93, 'subsample': 0.7670668119117988, 'colsample_bytree': 0.5739627990921417, 'reg_alpha': 1.4706547247718655, 'reg_lambda': 0.8259385167562873}
]

study = optuna.create_study(direction='maximize')
for params in known_candidates:
    study.enqueue_trial(params)

print(f"🚀 Running {len(known_candidates)} trials...")
study.optimize(objective, n_trials=len(known_candidates))

# ================= 6. 自动验证 (演示如何使用保存的索引) =================
print("\n" + "="*50)
print("🔍 SELF-VERIFICATION (Reloading Data via Indices)")
print("="*50)

# 1. 重新加载原始 Human 数据 (模拟新环境)
print("📥 Reloading Raw Human Data...")
df_human_raw = preprocess_human_data(PATH_HUMAN, PATH_HUMAN_TIME)

# 2. 加载保存的索引
print(f"📥 Loading Indices from {indices_save_path}...")
with open(indices_save_path, 'r') as f:
    indices = json.load(f)

test_idx = indices['human_test_idx'] # 获取测试集索引

# 3. 重构测试集
print("✂️ Reconstructing Test Set using Indices...")
df_test_reconstructed = df_human_raw.iloc[test_idx].reset_index(drop=True)

# 4. 加载特征顺序
with open(feature_save_path, 'r') as f:
    feature_cols = json.load(f)

X_test_recon = df_test_reconstructed[feature_cols]
y_test_recon = df_test_reconstructed['mean_te']

# 5. 加载刚才跑出来的某个模型验证
model_files = glob.glob(os.path.join(SAVE_DIR, "*.txt"))
if model_files:
    test_model_path = model_files[1] # 随便取一个，比如 Trial 1 (0.73)
    print(f"🔮 Testing verification on model: {os.path.basename(test_model_path)}")
    
    bst = lgbm.Booster(model_file=test_model_path)
    preds = bst.predict(X_test_recon.to_numpy())
    score = r2_score(y_test_recon, preds)
    
    print("   -> Held-out reconstruction verification completed.")
else:
    print("⚠️ No models found to verify.")
