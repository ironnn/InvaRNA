from typing import Union, List, Dict
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

# 假设这些是你项目中的组件
from src.data import RiboNNDataModule
from src.model import RiboNN
from src.utils.helpers import extract_config

def run_ribonn_prediction(
    inputs: Union[Dict, List[Dict], pd.DataFrame],
    run_df: pd.DataFrame,
    top_k: int = 5,
    batch_size: int = 1024,
    num_workers: int = 4
) -> pd.DataFrame:
    """
    高性能推理接口：兼容 Dict, List[Dict], DataFrame。
    优化点：模型预加载 + 显存内集成 + 自动生成 tx_id + 自动计算均值。
    """
    # --- 1. 数据归一化 ---
    if isinstance(inputs, dict):
        df = pd.DataFrame([inputs])
    elif isinstance(inputs, list):
        df = pd.DataFrame(inputs)
    else:
        df = inputs.copy()

    # 自动生成 tx_id（如果不存在）
    if "tx_id" not in df.columns:
        df["tx_id"] = [f"tx_{i:04d}" for i in range(len(df))]

    # --- 2. 环境配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 提取配置（假设从 run_df 的第一个记录提取基础架构）
    first_run_id = run_df.run_id.iloc[0]
    config = extract_config(run_df, first_run_id)
    config.update({
        "max_utr5_len": 1381, 
        "max_cds_utr3_len": 11937,
        "num_workers": num_workers, 
        "test_batch_size": batch_size,
        "pin_memory": True if torch.cuda.is_available() else False
    })

    # --- 3. 高效模型加载 (Top K 逻辑) ---
    # 根据验证集表现排序，选取前 K 个模型
    top_runs = run_df.sort_values("metrics.val_r2", ascending=False).head(top_k)
    models = []
    print(f"Loading top {len(top_runs)} models onto {device}...")
    for run_id in top_runs.run_id:
        model = RiboNN(**config)
        # 加载权重，map_location 防止在多显卡环境下出错
        state_dict_path = f"models/{run_id}/state_dict.pth"
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        model.to(device).eval()
        models.append(model)

    # --- 4. 数据模块初始化 ---
    dm = RiboNNDataModule(config, df=df)
    dm.setup("predict")
    
    # --- 5. 执行推理流水线 ---
    all_preds = []
    with torch.inference_mode(): # 比 no_grad 更快
        for batch in dm.predict_dataloader():
            batch = batch.to(device)
            # 核心优化：针对一个 batch，并行运行所有模型，并在 GPU 上直接平均
            # shape: [num_models, batch_size, num_tasks]
            batch_ensemble = torch.stack([m(batch) for m in models])
            mean_batch_pred = batch_ensemble.mean(dim=0) 
            all_preds.append(mean_batch_pred.cpu().numpy())

    # --- 6. 后处理与列名映射 ---
    final_preds = np.concatenate(all_preds, axis=0)
    
    # 细胞系名称列表 (截断显示，实际应用中建议从 config 或外部文件动态获取)
    raw_cols = "108T,12T,A2780,A549,BJ,BRx.142,C643,CRL.1634,Calu.3,Cybrid_Cells,H1.hESC,H1933,H9.hESC,HAP.1,HCC_tumor,HCC_adjancent_normal,HCT116,HEK293,HEK293T,HMECs,HSB2,HSPCs,HeLa,HeLa_S3,HepG2,Huh.7.5,Huh7,K562,Kidney_normal_tissue,LCL,LuCaP.PDX,MCF10A,MCF10A.ER.Src,MCF7,MD55A3,MDA.MB.231,MM1.S,MOLM.13,Molt.3,Mutu,OSCC,PANC1,PATU.8902,PC3,PC9,Primary_CD4._T.cells,Primary_human_bronchial_epithelial_cells,RD.CCL.136,RPE.1,SH.SY5Y,SUM159PT,SW480TetOnAPC,T47D,THP.1,U.251,U.343,U2392,U2OS,Vero_6,WI38,WM902B,WTC.11,ZR75.1,cardiac_fibroblasts,ccRCC,early_neurons,fibroblast,hESC,human_brain_tumor,iPSC.differentiated_dopamine_neurons,megakaryocytes,muscle_tissue,neuronal_precursor_cells,neurons,normal_brain_tissue,normal_prostate,primary_macrophages,skeletal_muscle"
    pred_col_names = [f"predicted_TE_{c.strip()}" for c in raw_cols.split(",")]
    
    res_df = pd.DataFrame(final_preds, columns=pred_col_names)
    
    # 计算均值：横向对所有细胞系预测值取平均
    res_df["mean_predicted_TE"] = res_df.mean(axis=1)
    
    # 组合输入和输出
    output_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
    
    return output_df

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 注意：运行前请确保你已经获取了 run_df (例如通过 mlflow.search_runs)
    # 这里假设 run_df 已经加载到内存中
    # run_df = mlflow.search_runs(experiment_ids=["..."]) 

    # 场景 1: 输入单条序列 (Dict)
    print("\n--- 场景 1: 单条序列 (Dict) ---")
    seq_dict = {
        "utr5_sequence": "GGGAAATAAGAGAGAAAAGAAGAG",
        "cds_sequence": "ATGAAATAG",
        "utr3_sequence": "TGAGCGGCCGC"
    }
    # 假设 run_df 已经在外部获取
    try:
        res_single = run_ribonn_prediction(seq_dict, run_df=run_df, top_k=5)
        print(res_single[["tx_id", "mean_predicted_TE"]])
    except NameError:
        print("跳过测试：未检测到有效 run_df 对象")

    # 场景 2: 输入多条序列 (List[Dict])
    print("\n--- 场景 2: 多条序列 (List[Dict]) ---")
    seq_list = [
        {
            "utr5_sequence": "GGGAG",
            "cds_sequence": "ATGAAATAG",
            "utr3_sequence": "TGA"
        },
        {
            "utr5_sequence": "AAAAA",
            "cds_sequence": "ATGCCTTAG",
            "utr3_sequence": "TAG"
        }
    ]
    try:
        res_list = run_ribonn_prediction(seq_list, run_df=run_df, top_k=5)
        print(res_list[["tx_id", "mean_predicted_TE"]])
    except NameError:
        print("跳过测试：未检测到有效 run_df 对象")

    # 场景 3: 输入多条序列 (DataFrame)
    print("\n--- 场景 3: 多条序列 (DataFrame) ---")
    data = {
        "utr5_sequence": ["GGGAG", "AAAAA"],
        "cds_sequence": ["ATGAAATAG", "ATGCCTTAG"],
        "utr3_sequence": ["TGA", "TAG"]
    }
    df_input = pd.DataFrame(data)
    try:
        res_df = run_ribonn_prediction(df_input, run_df=run_df, top_k=5)
        print(res_df[["tx_id", "mean_predicted_TE"]])
    except NameError:
        print("跳过测试：未检测到有效 run_df 对象")