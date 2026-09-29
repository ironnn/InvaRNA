import os
import json
import joblib
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ================= 1. 全局配置 =================
# ⚡️ 并行配置
os.environ['OMP_NUM_THREADS'] = '80'  # 控制 OpenMP 线程数
N_JOBS = 160

# 📂 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    'mouse_time': os.path.join(SCRIPT_DIR, "data/input/mouse_time.csv"),
    'human_time': os.path.join(SCRIPT_DIR, "data/input/human_time.csv"),
    'human_parquet': os.path.join(SCRIPT_DIR, "data/input/train_ready_humanmutation_region_20260118_pred_8gpu_bf16.parquet"),
    'mouse_parquet': os.path.join(SCRIPT_DIR, "data/input/train_ready_mousemutation_region_20260118_pred_8gpu_bf16.parquet"),
    # Teacher 模型
    'model': os.path.join(SCRIPT_DIR, "data/input/trial_2_TestR2_0.73042.pkl"),
    # 索引 (最重要！)
    'indices': os.path.join(SCRIPT_DIR, "data/input/indices/split_indices.json"),
    # 原始数据 (用于获取 Ground Truth)
    'human_orig': os.path.join(SCRIPT_DIR, "data/input/train_ready_human_regional_strict_20260117_pred_8gpu_bf16.pkl"),
    # 输出
    'save_final': os.path.join(SCRIPT_DIR, "data/output/student_train_merged_final_v3.parquet"),
}

TARGET_R2 = 0.73042  # 目标对齐分数

# ================= 2. 核心组件函数 =================

def get_adaptive_weight(steps, switch_point=80, k=0.1):
    return 1 / (1 + np.exp(k * (steps - switch_point)))

def preprocess_hl(df, time_path):
    """映射 Half-Life，确保特征一致性"""
    print(f"   [Preprocess] Mapping Half-Life from {os.path.basename(time_path)}...")
    df_time = pd.read_csv(time_path)
    hl_map = df_time.drop_duplicates('Ensembl Gene Id').set_index('Ensembl Gene Id')['half-life (PC1)']
    
    is_mut0 = df['transcript_id'].astype(str).str.endswith('mut0')
    clean_gene_ids = df.loc[is_mut0, 'gene_id'].astype(str).str.split('.').str[0]
    mapped_vals = clean_gene_ids.map(hl_map)
    
    # 填充逻辑
    df['hl'] = df['pred_score']
    df.loc[is_mut0, 'hl'] = mapped_vals.fillna(df.loc[is_mut0, 'hl'])
    
    return df.drop(columns=['pred_score'], errors='ignore')

# ... (前面的配置不变) ...

# ================= 3. 验证与映射构建 (Gene ID Version) =================

def build_map_and_verify(df_human_new, model, feature_names):
    print("\n" + "="*50)
    print("🔒 STEP 1: VERIFICATION & INDEX MAPPING (BY human_gene_id)")
    print("="*50)

    # --- A. 加载原始索引与数据 ---
    with open(PATHS['indices'], 'r') as f:
        indices = json.load(f)
    
    # [正确] 读取 human_gene_id
    print("   - Loading Source Data (cols: human_gene_id, transcript_id, humanTE)...")
    df_orig = pd.read_pickle(PATHS['human_orig'])[['human_gene_id', 'transcript_id', 'humanTE']].reset_index(drop=True)
    
    # --- B. 构建 Split Map ---
    print("   - Building Gene-based Split Map (Safe Cleaning)...")
    split_map = {}
    
    # [修正] 定义清洗函数：转字符串 -> 去版本号 -> 去空格
    # 这是为了确保和 Parquet 里的 gene_id 100% 能对上
    def clean_ensg(series):
        return series.astype(str).str.split('.').str[0].str.strip().values

    train_genes = clean_ensg(df_orig.iloc[indices['human_train_idx']]['human_gene_id'])
    val_genes   = clean_ensg(df_orig.iloc[indices['human_val_idx']]['human_gene_id'])
    test_genes  = clean_ensg(df_orig.iloc[indices['human_test_idx']]['human_gene_id'])
    
    split_map.update(dict.fromkeys(train_genes, 'train'))
    split_map.update(dict.fromkeys(val_genes, 'val'))
    split_map.update(dict.fromkeys(test_genes, 'test'))
    
    print(f"   (Mapped {len(split_map)} unique genes)")

    # --- C. 准备 GT ---
    orig_test = df_orig.iloc[indices['human_test_idx']]
    orig_test['clean_tid'] = orig_test['transcript_id'].astype(str).str.split('_').str[0].str.split('.').str[0]
    test_gt_map = dict(zip(orig_test['clean_tid'], orig_test['humanTE']))
    
    del df_orig, indices
    gc.collect()

    # --- D. 在新数据上验证 (必须用到 split_map!) ---
    print("   - Verifying on NEW Parquet data...")
    
    df_wt_new = df_human_new[df_human_new['transcript_id'].astype(str).str.endswith('mut0')].copy()
    
    # 1. [关键] 在验证时，必须清洗 gene_id 并尝试 map
    # 如果这里 map 失败，verify 就会报错，我们就知道问题出在 ID 格式上，而不是等到跑完才发现全是 train
    df_wt_new['clean_gene_id'] = df_wt_new['gene_id'].astype(str).str.split('.').str[0].str.strip()
    df_wt_new['split_check'] = df_wt_new['clean_gene_id'].map(split_map)
    
    # 2. 筛选 Test Set
    df_verify = df_wt_new[df_wt_new['split_check'] == 'test'].copy()
    
    if len(df_verify) == 0:
        print("❌ ERROR: Map verification failed! 'split_map' returned 0 test samples.")
        print("   This means Gene IDs in Parquet do not match Gene IDs in Source.")
        return split_map

    # 3. 匹配 GT 并算分
    df_verify['clean_tid'] = df_verify['transcript_id'].astype(str).str.split('_').str[0].str.split('.').str[0]
    df_verify['y_true'] = df_verify['clean_tid'].map(test_gt_map)
    df_verify.dropna(subset=['y_true'], inplace=True)

    print(f"     Verifying on {len(df_verify)} test samples...")
    X_verify = df_verify[feature_names].values.astype(np.float32)
    try: preds = model.predict(X_verify, num_threads=N_JOBS)
    except: preds = model.predict(X_verify)
    
    score = r2_score(df_verify['y_true'], preds)
    print(f"   - Target R2 : {TARGET_R2:.5f}")
    print(f"   - Actual R2 : {score:.5f}")
    
    if abs(score - TARGET_R2) < 0.001:
        print("✅ SUCCESS: Data Alignment Verified.")
    else:
        print("⚠️ WARNING: R2 Mismatch!")
        
    return split_map
# ================= 4. 处理流水线 (Gene ID Mapping) =================

def process_species(df, species, model, feature_names, split_map=None):
    print(f"\n🦕 Processing {species}...")
    
    # 1. 预测
    X = df[feature_names].values.astype(np.float32)
    try: preds = model.predict(X, num_threads=N_JOBS)
    except: preds = model.predict(X)
    df['pred_score'] = preds
    
    # 2. 构建 Anchor
    try:
        df['mut_step'] = df['transcript_id'].astype(str).str.split('mut').str[-1].astype(int).clip(upper=200)
    except:
        df['mut_step'] = 200
        df.loc[df['transcript_id'].str.endswith('mut0'), 'mut_step'] = 0

    # 这里的 clean_id 仅用于内部匹配 WT anchor，transcript level
    df['clean_tid'] = df['transcript_id'].astype(str).str.split('_').str[0].str.split('.').str[0]
    
    df_wt = df[df['mut_step'] == 0]
    col_real = 'mean_te' if 'mean_te' in df_wt.columns else 'hl'
    
    d_pred = dict(zip(df_wt['clean_tid'], df_wt['pred_score']))
    d_real = dict(zip(df_wt['clean_tid'], df_wt[col_real]))
    
    df['anchor_pred_wt'] = df['clean_tid'].map(d_pred)
    df['anchor_real_wt'] = df['clean_tid'].map(d_real)
    df.dropna(subset=['anchor_pred_wt', 'anchor_real_wt'], inplace=True)
    
    # 3. Soft label 计算已拆分到 step9 的独立脚本中
    #    anchor_pred_wt / anchor_real_wt 保留在输出中供下游使用

    # 4. 打标 (Using GENE ID)
    df['species'] = species
    if species == 'human':
        if split_map is None: raise ValueError("Human requires split_map!")
        
        # === 核心修改：使用 gene_id 匹配 ===
        # 假设 gene_id 列存在且干净
        print("   Using 'gene_id' for splitting...")
        # [修改后] ✅ 强制清洗，确保 100% 匹配
        df['clean_gene_id'] = df['gene_id'].astype(str).str.split('.').str[0].str.strip()
        df['split'] = df['clean_gene_id'].map(split_map)
        df.drop(columns=['clean_gene_id'], inplace=True) # 用完删掉
        
        missing = df['split'].isna().sum()
        if missing > 0:
            print(f"   ⚠️ Warning: {missing} human sequences did not match any gene in split_map.")
            # 打印几个没匹配上的 gene_id 看看
            print(f"      Example missing gene_id: {df.loc[df['split'].isna(), 'gene_id'].head(1).values}")
        
        df['split'] = df['split'].fillna('train')
        
    else:
        # Mouse: All Train
        df['split'] = 'train'
        
    df['data_type'] = df['split'].astype(str) + '_' + np.where(df['mut_step']==0, 'wt', 'mut')
    df['species'] = df['species'].astype('category')
    df['data_type'] = df['data_type'].astype('category')
    
    # 5. 清理 (保留 anchor 列供 step9 使用)
    cols_to_drop = ['clean_tid', 'split']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    return df

# ================= 5. 主程序 =================

if __name__ == "__main__":
    print("🚀 Pipeline Started...")
    
    # 1. 加载模型
    model = joblib.load(PATHS['model'])
    feat_names = model.feature_name() if hasattr(model, 'feature_name') else model.booster_.feature_name()
    
    # 2. 加载 Human 数据 (必须先加载来做验证)
    print("📂 Loading Human Parquet...")
    df_human = pd.read_parquet(PATHS['human_parquet'])
    df_human = preprocess_hl(df_human, PATHS['human_time']) # 先映射HL，否则预测不准
    
    # 3. 验证并获取 Map (核心步骤)
    # 传入刚刚加载并预处理好的 df_human
    human_split_map = build_map_and_verify(df_human, model, feat_names)
    
    # 4. 处理 Human (沿用 Map)
    print("   (Processing Human using verified map...)")
    df_human = process_species(df_human, 'human', model, feat_names, split_map=human_split_map)
    
    # 5. 处理 Mouse
    print("📂 Loading Mouse Parquet...")
    df_mouse = pd.read_parquet(PATHS['mouse_parquet'])
    df_mouse = preprocess_hl(df_mouse, PATHS['mouse_time'])
    df_mouse = process_species(df_mouse, 'mouse', model, feat_names)
    
    # 6. 合并保存
    print("🔗 Merging...")
    df_final = pd.concat([df_human, df_mouse], axis=0, ignore_index=True)
    df_final.to_parquet(PATHS['save_final'], index=False)
    
    print(f"✅ DONE. Saved to {PATHS['save_final']}")
    print(df_final.groupby(['species', 'data_type']).size())
