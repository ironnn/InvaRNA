import lightgbm as lgbm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import joblib

# ================= 1. 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 111
# 🎯 指定你要验证的那个神级模型路径
MODEL_PATH = os.path.join(SCRIPT_DIR, "data/input/trial_2_TestR2_0.73042.pkl")

# 数据路径
PATH_HUMAN = os.path.join(SCRIPT_DIR, "data/input/train_ready_human_regional_strict_20260117_pred_8gpu_bf16.pkl")
PATH_MOUSE = os.path.join(SCRIPT_DIR, "data/input/train_ready_mouse_regional_strict_20260117_pred_8gpu_bf16.pkl")
PATH_MOUSE_TIME = os.path.join(SCRIPT_DIR, "data/input/mouse_time.csv")
PATH_HUMAN_TIME = os.path.join(SCRIPT_DIR, "data/input/human_time.csv")

# ================= 2. 数据准备 (必须完全一致) =================
print(f"🚀 Starting Verification for: {MODEL_PATH}")

def preprocess_human_data(path, path_time):
    print(f"Loading Human data from {path}...")
    df = pd.read_pickle(path)
    df_time = pd.read_csv(path_time)
    df['hl'] = df['pred_score']
    if 'humanTE' in df.columns: df.rename(columns={'humanTE': 'mean_te'}, inplace=True)
    clean_gene_ids = df['human_gene_id'].astype(str).str.split('.').str[0]
    mapping_series = df_time.drop_duplicates(subset=['Ensembl Gene Id']).set_index('Ensembl Gene Id')['half-life (PC1)']
    df['hl'] = clean_gene_ids.map(mapping_series).fillna(df['hl'])
    return df

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
    return df

def get_sorted_feature_cols(all_columns, ignore_list):
    feat_cols = [c for c in all_columns if c not in ignore_list]
    emb_cols = [c for c in feat_cols if c.startswith('feat_')]
    hl_cols_found = [c for c in feat_cols if 'halflife' in c.lower() or 'hl' in c.lower()]
    hand_cols = [c for c in feat_cols if (c not in emb_cols) and (c not in hl_cols_found)]
    final_order = emb_cols + hand_cols + hl_cols_found
    print(f"🔒 Feature Order Locked: {len(final_order)}")
    return final_order

# 加载数据
df_human = preprocess_human_data(PATH_HUMAN, PATH_HUMAN_TIME)
df_mouse = preprocess_mouse_data(PATH_MOUSE, PATH_MOUSE_TIME)
df_human['species'] = 'human'
df_mouse['species'] = 'mouse'

# 切分 (严格保证 SEED=111)
print("\n✂️ Re-creating Data Splits (SEED=111)...")
train_df_human, temp_df = train_test_split(df_human, test_size=0.2, random_state=SEED, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, shuffle=True)
train_df_mouse = df_mouse.copy()

# 混合与特征锁定
print("\n🔄 Preparing Features...")
common_cols = [c for c in train_df_human.columns if c in train_df_mouse.columns]
train_df_mixed = pd.concat([train_df_human[common_cols], train_df_mouse[common_cols]], axis=0).reset_index(drop=True)

IGNORE_COLS = ['transcript_id', 'gene_id', 'mean_te', 'mouseTE', 'te', 
               'mrna', 'utr5_size', 'cds_size', 'utr3_size', 'mouse_half_life', 
               'pred_score', 'species'] 

sota_feature_order = get_sorted_feature_cols(train_df_mixed.columns, IGNORE_COLS)

# 只需要准备 X_test
X_test = test_df[sota_feature_order]
y_test = test_df['mean_te']

# ================= 3. 加载模型与验证 =================
print(f"\n📂 Loading Model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# 加载 .pkl 文件
model = joblib.load(MODEL_PATH)

print("🔮 Running Inference on Test Set...")
preds = model.predict(X_test)
score = r2_score(y_test, preds)

print("\n" + "="*50)
print(f"🏆 VERIFICATION RESULT")
print("="*50)
print(f"Model File: {os.path.basename(MODEL_PATH)}")
print(f"Test R2:    {score:.5f}")

if score > 0.729:
    print("\n✅ PERFECT! The 0.73 result is confirmed.")
    print("   You can now safely use this model for distillation.")
else:
    print(f"\n⚠️ WARNING: Score is {score:.5f}. Please check data consistency.")


import matplotlib.pyplot as plt
import seaborn as sns

# ================= 4. 特征重要性分析 (Gain) =================
print("\n📊 Extracting Feature Importance (Gain)...")

# 1. 确保拿到特征名 (从模型 Booster 中拿最准确)
# 因为 LightGBM 训练时记录了特征名，直接取比用 X_test.columns 更稳
feature_names = model.booster_.feature_name()

# 2. 获取 Gain 和 Split
importance_gain = model.booster_.feature_importance(importance_type='gain')
importance_split = model.booster_.feature_importance(importance_type='split')

# 3. 构建 DataFrame
feature_imp = pd.DataFrame({
    'feature': feature_names,
    'gain': importance_gain,
    'split': importance_split
})

# 4. 按 Gain 降序排列
top_features = feature_imp.sort_values(by='gain', ascending=False).reset_index(drop=True)

# 5. 打印 Top 30
print("\n🏆 TOP 30 Important Features (by Gain):")
print("-" * 60)
print(f"{'Rank':<5} {'Feature':<30} {'Gain':<15} {'Split':<10}")
print("-" * 60)

for i in range(30):
    row = top_features.iloc[i]
    print(f"{i+1:<5} {row['feature']:<30} {row['gain']:.2f}           {int(row['split'])}")
print("-" * 60)

# ================= 5. 可视化 Top 20 =================
plt.figure(figsize=(12, 10))
sns.barplot(x="gain", y="feature", data=top_features.head(20), palette="viridis")
plt.title('LightGBM Feature Importance (Gain) - Top 20', fontsize=15)
plt.xlabel('Total Gain (Importance)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.tight_layout()

# 保存图片
img_path = os.path.join(SCRIPT_DIR, "data/output", "feature_importance_gain.png")
plt.savefig(img_path)
print(f"\n🖼️ Feature importance plot saved to: {img_path}")
plt.show()

# ================= 6. 深度解读建议 =================
print("\n💡 观察重点:")
print("1. hl (Half-Life) 是否在 Top 10？ -> 如果在，说明 Teacher 成功利用了特权信息。")
print("2. cds_log_size 是否是 Top 1？ -> 符合生物学规律（长度决定稳定性）。")
print("3. feat_xxxx (Embedding) 有多少进入 Top 20？ -> 越多说明 Embedding 提取到了深层 Motif。")
