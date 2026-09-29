import sys
import os
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
from pathlib import Path

# ================= 导入 RiboNN 预测函数 =================
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from predict_ribonn import run_ribonn_prediction

# ================= 0. 分布式环境初始化 =================
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, world_size, device
    else:
        print("⚠️ 未检测到分布式环境，回退到单卡模式 (CUDA:0)")
        return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RANK, WORLD_SIZE, DEVICE = setup_distributed()

def log(msg):
    if RANK == 0:
        print(f"[Rank {RANK}] {msg}")

# ================= 1. 路径配置 =================
# 数据集路径
TEST_DATA_PATH = 'NEEDS_INPUT_TEST_PARQUET'
MUTANT_DATA_PATH = "NEEDS_INPUT_MUTANT_TABLE"

# 输出路径
FINAL_OUTPUT_CSV = "outputs/ribonn_predictions.csv"

# ================= 2. 序列切分函数 =================
def split_mrna_sequence(row):
    """
    从 mrna 和各个 size 字段切分出 UTR5, CDS, UTR3
    """
    mrna = str(row['mrna'])
    utr5_size = int(row['utr5_size'])
    cds_size = int(row['cds_size'])
    utr3_size = int(row['utr3_size'])

    utr5_sequence = mrna[0:utr5_size]
    cds_sequence = mrna[utr5_size:utr5_size+cds_size]
    utr3_sequence = mrna[utr5_size+cds_size:utr5_size+cds_size+utr3_size]

    return pd.Series({
        'utr5_sequence': utr5_sequence,
        'cds_sequence': cds_sequence,
        'utr3_sequence': utr3_sequence
    })

# ================= 3. 通用推理核心函数 =================
def run_distributed_inference(df_data, phase_name="", n_folds=5):
    """
    通用的分布式推理函数，使用 RiboNN 预测
    返回: 如果是 Rank 0，返回汇总后的 DataFrame (包含 'pred' 列); 其他 Rank 返回 None
    """
    # 0. 检查并切分序列（如果需要）
    required_cols = ['utr5_sequence', 'cds_sequence', 'utr3_sequence']
    missing_cols = [col for col in required_cols if col not in df_data.columns]

    if missing_cols:
        if RANK == 0:
            print(f"   检测到缺失列 {missing_cols}，开始从 mrna 切分序列...")
        # 需要切分序列
        seq_cols = ['mrna', 'utr5_size', 'cds_size', 'utr3_size']
        if not all(col in df_data.columns for col in seq_cols):
            raise ValueError(f"无法切分序列，缺少必需的列。需要: {seq_cols}")

        seq_df = df_data.apply(split_mrna_sequence, axis=1)
        df_data = pd.concat([df_data, seq_df], axis=1)

        if RANK == 0:
            print(f"   ✅ 序列切分完成")

    # 1. 切分数据
    # 确保 reset_index，以便后续排序恢复
    df_sharded = df_data.reset_index(drop=True)
    my_shard = df_sharded.iloc[RANK::WORLD_SIZE].copy()

    if RANK == 0:
        print(f"\n🚀 [{phase_name}] Start Inference. Total: {len(df_data)} | Per GPU: ~{len(my_shard)}")

    # 2. 推理循环 - 使用字典来存储预测结果，通过索引匹配
    batch_size = 32
    pred_dict = {}  # {原始索引: 预测值}

    iterator = range(0, len(my_shard), batch_size)
    if RANK == 0:
        iterator = tqdm(iterator, desc=f"[{phase_name}] Running", total=len(iterator))

    for i in iterator:
        batch_df = my_shard.iloc[i : i+batch_size].copy()
        batch_indices = batch_df.index.tolist()  # 记录原始索引

        # 添加 tx_id 用于后续匹配（使用索引作为 ID）
        batch_df['tx_id'] = [f"idx_{idx}" for idx in batch_indices]

        # 调用 RiboNN 预测函数
        try:
            batch_pred = run_ribonn_prediction(batch_df, n_folds=n_folds)

            # 通过 tx_id 匹配结果
            for _, row in batch_pred.iterrows():
                tx_id = row['tx_id']
                if tx_id.startswith('idx_'):
                    original_idx = int(tx_id.split('_')[1])
                    pred_dict[original_idx] = row['mean_predicted_TE']
        except Exception as e:
            print(f"[Rank {RANK}] Error in batch {i}: {e}")
            import traceback
            traceback.print_exc()

    # 3. 将预测结果按索引顺序填充（缺失的用 NaN）
    preds = [pred_dict.get(idx, np.nan) for idx in my_shard.index]
    my_shard['pred'] = preds
    temp_file = f"temp_{phase_name}_rank_{RANK}.parquet"
    my_shard.to_parquet(temp_file)

    # 4. 同步
    if WORLD_SIZE > 1:
        dist.barrier()

    # 5. 汇总结果 (Rank 0)
    df_final = None
    if RANK == 0:
        print(f"📥 [{phase_name}] Merging results...")
        df_list = []
        for r in range(WORLD_SIZE):
            fname = f"temp_{phase_name}_rank_{r}.parquet"
            try:
                df_part = pd.read_parquet(fname)
                df_list.append(df_part)
                os.remove(fname)
            except Exception as e:
                print(f"❌ Error reading {fname}: {e}")

        if df_list:
            df_final = pd.concat(df_list, axis=0)
            df_final = df_final.sort_index() # 恢复原始顺序
            print(f"✅ [{phase_name}] Done. Shape: {df_final.shape}")

    #再次同步确保Rank0处理完文件
    if WORLD_SIZE > 1:
        dist.barrier()

    return df_final

# ================= 4. 主程序 =================
def main():
    # ================= 阶段 1: 验证 Test Set R2 =================
    log("🔹 Step 1: Loading Test Data for R2 Validation...")
    df_test = pd.read_parquet(TEST_DATA_PATH)

    # 运行推理（内部会自动切分序列）
    df_test_result = run_distributed_inference(df_test, phase_name="TEST_R2", n_folds=1)

    if RANK == 0:
        # 计算 R2
        # 假设真实标签列名为 'mean_te' 或类似
        if 'mean_te' in df_test_result.columns:
            label_col = 'mean_te'
        elif 'TE' in df_test_result.columns:
            label_col = 'TE'
        else:
            # 尝试查找包含 'te' 的列
            label_candidates = [c for c in df_test_result.columns if 'te' in c.lower()]
            if label_candidates:
                label_col = label_candidates[0]
                print(f"⚠️ 使用 {label_col} 作为真实标签列")
            else:
                print("⚠️ 警告: 未找到真实标签列，跳过 R2 计算")
                label_col = None

        if label_col:
            y_true = df_test_result[label_col].values
            y_pred = df_test_result['pred'].values
            # 过滤掉 NaN 值
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid_mask.sum() > 0:
                score = r2_score(y_true[valid_mask], y_pred[valid_mask])

                print("\n" + "="*40)
                print(f"🏆 Validation Result on Test Lite")
                print(f"   R2 Score: {score:.4f}")
                print(f"   Valid samples: {valid_mask.sum()} / {len(y_true)}")
                print("="*40 + "\n")

                if score < 0.6:
                    print("⚠️ 警告: R2 分数显著低于预期，请检查模型或数据！")
            else:
                print("❌ 错误: 没有有效的样本进行 R2 计算")

    # ================= 阶段 2: 推理突变体数据 =================
    # 清理内存
    del df_test
    if RANK == 0 and df_test_result is not None:
        del df_test_result
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    log("🔹 Step 2: Preparing Mutant Data...")

    # 1. 读取原始数据
    df_combined = pd.read_pickle(MUTANT_DATA_PATH)

    # 2. 读取Test集用于筛选ID
    df_test_ids = pd.read_parquet(TEST_DATA_PATH, columns=['gene_id'])

    # 3. 筛选数据
    # 假设 df_combined 中有 'human_gene_id' 列
    if 'human_gene_id' in df_combined.columns:
        df_test_mut = df_combined[df_combined['human_gene_id'].isin(df_test_ids['gene_id'].unique())].copy()
    else:
        print("⚠️ 警告: df_combined 中没有 'human_gene_id' 列，使用全部数据")
        df_test_mut = df_combined.copy()

    # 运行推理（内部会自动切分序列）
    df_mut_result = run_distributed_inference(df_test_mut, phase_name="MUTANT_INFER", n_folds=1)

    if RANK == 0:
        print(f"💾 Saving final results to {FINAL_OUTPUT_CSV}...")
        df_mut_result.to_csv(FINAL_OUTPUT_CSV, index=False)
        print("🎉 All Tasks Completed Successfully!")

    if WORLD_SIZE > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
