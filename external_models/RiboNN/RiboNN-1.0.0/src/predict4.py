"""
终极优化预测模块 - predict4.py
核心突破：反转循环顺序，从 "for model -> for batch" 改为 "for batch -> for model"

性能提升：
1. DataLoader只运行1次（不是N次）
2. 数据只从内存搬运到GPU 1次（不是N次）
3. 预处理逻辑只执行1次（不是N次）
4. 支持模型批量加载或分批加载（显存权衡）

对于5个模型：理论加速 ~5x（数据加载部分）
"""

from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import torch

from src.data import RiboNNDataModule
from src.model import RiboNN
from src.utils.helpers import extract_config

pd.set_option('use_numexpr', False)


def predict_using_models_trained_in_one_fold(
    run_df: pd.DataFrame,
    config: Dict,
    dm,  # RiboNNDataModule
    top_k_models_to_use: int = 1,
    use_amp: bool = False,
    device: Optional[torch.device] = None,
    max_models_in_memory: Optional[int] = None,  # 新增：控制同时加载的模型数
) -> pd.DataFrame:
    """
    终极优化版本：反转循环顺序，极致减少数据搬运

    关键优化：
    - 循环反转：for batch -> for model（而不是 for model -> for batch）
    - 数据只加载1次，所有模型复用同一批数据
    - 支持显存控制：max_models_in_memory 参数

    Args:
        max_models_in_memory: 同时加载到GPU的最大模型数
            - None: 加载所有模型（最快，但需要更多显存）
            - int: 分批加载模型（显存友好）
    """
    # TE names
    predicted_columns = "TE_108T,TE_12T,TE_A2780,TE_A549,TE_BJ,TE_BRx.142,TE_C643,TE_CRL.1634,TE_Calu.3,TE_Cybrid_Cells,TE_H1.hESC,TE_H1933,TE_H9.hESC,TE_HAP.1,TE_HCC_tumor,TE_HCC_adjancent_normal,TE_HCT116,TE_HEK293,TE_HEK293T,TE_HMECs,TE_HSB2,TE_HSPCs,TE_HeLa,TE_HeLa_S3,TE_HepG2,TE_Huh.7.5,TE_Huh7,TE_K562,TE_Kidney_normal_tissue,TE_LCL,TE_LuCaP.PDX,TE_MCF10A,TE_MCF10A.ER.Src,TE_MCF7,TE_MD55A3,TE_MDA.MB.231,TE_MM1.S,TE_MOLM.13,TE_Molt.3,TE_Mutu,TE_OSCC,TE_PANC1,TE_PATU.8902,TE_PC3,TE_PC9,TE_Primary_CD4._T.cells,TE_Primary_human_bronchial_epithelial_cells,TE_RD.CCL.136,TE_RPE.1,TE_SH.SY5Y,TE_SUM159PT,TE_SW480TetOnAPC,TE_T47D,TE_THP.1,TE_U.251,TE_U.343,TE_U2392,TE_U2OS,TE_Vero_6,TE_WI38,TE_WM902B,TE_WTC.11,TE_ZR75.1,TE_cardiac_fibroblasts,TE_ccRCC,TE_early_neurons,TE_fibroblast,TE_hESC,TE_human_brain_tumor,TE_iPSC.differentiated_dopamine_neurons,TE_megakaryocytes,TE_muscle_tissue,TE_neuronal_precursor_cells,TE_neurons,TE_normal_brain_tissue,TE_normal_prostate,TE_primary_macrophages,TE_skeletal_muscle"
    predicted_columns = predicted_columns.replace("TE_", "predicted_TE_").split(",")

    # Filter run_df to keep the top k models ranked by validation R2
    run_df = run_df.sort_values("metrics.val_r2", ascending=False).head(
        top_k_models_to_use
    )

    # 设备设置
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取所有模型的run_id
    run_ids = run_df.run_id.tolist()
    num_models = len(run_ids)

    # 决定是否分批加载模型
    if max_models_in_memory is None:
        max_models_in_memory = num_models  # 一次性加载所有模型

    # ========== 核心优化：反转循环顺序 ==========

    # 获取dataloader（只创建一次！）
    dataloader = dm.predict_dataloader()

    # 预分配结果存储：[num_samples, num_features, num_models]
    # 先不知道样本数，用列表收集每个batch的结果
    all_batch_predictions = []  # 每个元素是 [batch_size, num_features, num_models]

    # 分批加载模型
    for model_batch_start in range(0, num_models, max_models_in_memory):
        model_batch_end = min(model_batch_start + max_models_in_memory, num_models)
        current_run_ids = run_ids[model_batch_start:model_batch_end]

        # 加载当前批次的所有模型
        models = []
        for run_id in current_run_ids:
            model = RiboNN(**config)
            local_state_dict_path = f"models/{run_id}/state_dict.pth"
            model.load_state_dict(torch.load(local_state_dict_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)

        # 重置dataloader（如果不是第一批模型）
        if model_batch_start > 0:
            dataloader = dm.predict_dataloader()

        # 遍历数据（只遍历一次！）
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(device)

                # 让当前批次的所有模型都预测这个batch
                batch_model_predictions = []

                for model in models:
                    # 可选：使用混合精度推理
                    if use_amp and torch.cuda.is_available():
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            pred = model(batch)
                    else:
                        pred = model(batch)

                    # 转为FP32并移到CPU
                    batch_model_predictions.append(pred.float().cpu())

                # 拼接当前batch的所有模型预测: [batch_size, num_features, num_models_in_current_batch]
                batch_model_predictions = torch.stack(batch_model_predictions, dim=-1)

                # 如果是第一批模型，创建新的存储；否则拼接到现有预测
                if model_batch_start == 0:
                    if batch_idx >= len(all_batch_predictions):
                        all_batch_predictions.append(batch_model_predictions)
                    else:
                        all_batch_predictions[batch_idx] = batch_model_predictions
                else:
                    # 拼接到已有的模型预测上
                    all_batch_predictions[batch_idx] = torch.cat(
                        [all_batch_predictions[batch_idx], batch_model_predictions], dim=-1
                    )

        # 清理当前批次的模型
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 拼接所有batch: [total_samples, num_features, num_models]
    all_predictions = torch.cat(all_batch_predictions, dim=0)

    # 平均所有模型的预测
    mean_prediction = all_predictions.mean(dim=-1).numpy()

    # 构建DataFrame
    df = pd.DataFrame(mean_prediction, columns=predicted_columns)
    df = pd.concat([dm.df, df], axis=1)

    return df


def predict_using_nested_cross_validation_models(
    run_df: pd.DataFrame,
    top_k_models_to_use: int = 5,
    batch_size: int = 2048,
    num_workers: int = 4,
    input_df: Optional[pd.DataFrame] = None,
    use_amp: bool = False,
    max_models_in_memory: Optional[int] = None,  # 新增
) -> pd.DataFrame:
    """
    终极优化版本：使用嵌套交叉验证模型进行预测

    主要优化：
    - 反转循环顺序（for batch -> for model）
    - 数据只加载1次
    - 支持显存控制

    Args:
        run_df: MLflow runs DataFrame
        top_k_models_to_use: 每个fold使用top-k个模型
        batch_size: 批处理大小（增大以提速）
        num_workers: DataLoader工作进程数
        input_df: 输入数据DataFrame
        use_amp: 是否使用自动混合精度（FP16）
        max_models_in_memory: 同时加载到GPU的最大模型数
            - None: 加载所有模型（最快，需要更多显存）
            - 1: 逐个加载模型（最省显存，但会退化为原始循环）
            - 2-4: 折中方案

    Returns:
        包含预测结果的DataFrame
    """

    # Create data module (只创建一次)
    config = extract_config(run_df, run_df.run_id.iloc[0])
    config["max_utr5_len"] = 1_381
    config["max_cds_utr3_len"] = 11_937
    config["num_workers"] = num_workers
    config["test_batch_size"] = batch_size
    config["remove_extreme_txs"] = False
    config["target_column_pattern"] = None

    dm = RiboNNDataModule(config, df=input_df)

    # 预分配列表
    all_prediction_dfs = []

    # 获取所有unique test folds
    test_folds = np.sort(run_df["params.test_fold"].unique())

    for test_fold in test_folds:
        test_fold_str = str(test_fold)
        sub_run_df = run_df.query(
            "`params.test_fold` == @test_fold_str or `params.test_fold` == @test_fold"
        ).reset_index(drop=True)

        prediction_df = predict_using_models_trained_in_one_fold(
            sub_run_df, config, dm, top_k_models_to_use,
            use_amp=use_amp,
            max_models_in_memory=max_models_in_memory
        )
        prediction_df["fold"] = int(test_fold)

        all_prediction_dfs.append(prediction_df)

    # 一次性合并所有fold的预测
    all_predictions = pd.concat(all_prediction_dfs, axis=0, ignore_index=True)

    return all_predictions


# ========== 兼容性函数 ==========

def predict_using_models_trained_in_one_fold_legacy(
    run_df: pd.DataFrame,
    config: Dict,
    dm,
    top_k_models_to_use: int = 5,
) -> pd.DataFrame:
    """
    原版兼容函数（调用优化版本，use_amp=False, 一次性加载所有模型）
    """
    return predict_using_models_trained_in_one_fold(
        run_df, config, dm, top_k_models_to_use,
        use_amp=False,
        max_models_in_memory=None
    )


def predict_using_nested_cross_validation_models_legacy(
    run_df: pd.DataFrame,
    top_k_models_to_use: int = 5,
    batch_size: int = 1024,
    num_workers: int = 4,
    input_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    原版兼容函数（调用优化版本，use_amp=False）
    """
    return predict_using_nested_cross_validation_models(
        run_df, top_k_models_to_use, batch_size, num_workers, input_df,
        use_amp=False,
        max_models_in_memory=None
    )
