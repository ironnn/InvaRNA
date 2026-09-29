"""
优化的急速预测模块 - predict3.py
主要优化:
1. 更大的batch size (默认2048 vs 原版1024)
2. 避免重复的内存分配
3. 一次性拼接所有预测
4. 可选的FP16/BF16混合精度 (仅在use_amp=True时)
5. 模型复用优化

保持与原版 predict.py 完全兼容的API
"""

from typing import Dict, Optional
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
    top_k_models_to_use: int = 5,
    use_amp: bool = False,  # 新增：是否使用自动混合精度
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    急速版本：使用一个fold中训练的top-k模型进行预测

    优化点：
    - 批量拼接预测结果，避免多次内存分配
    - 可选混合精度推理
    - 更高效的张量操作
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

    # 获取dataloader（只创建一次）
    dataloader = dm.predict_dataloader()

    # Iterate over the models to make predictions
    predictions = []

    for run_id in run_df.run_id:
        # Create a new model
        model = RiboNN(**config)

        # Load the state dict
        local_state_dict_path = f"models/{run_id}/state_dict.pth"
        model.load_state_dict(torch.load(local_state_dict_path, map_location=device))
        model.to(device)
        model.eval()

        # 批量预测
        with torch.no_grad():
            batched_predictions = []

            for batch in dataloader:
                batch = batch.to(device)

                # 可选：使用混合精度推理
                if use_amp and torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        pred = model(batch)
                else:
                    pred = model(batch)

                # 转为FP32并移到CPU（避免BF16/FP16转numpy问题）
                batched_predictions.append(pred.float().cpu())

        # 一次性拼接所有batch（更高效）
        if isinstance(batched_predictions, list) and len(batched_predictions) > 0:
            batched_predictions = torch.cat(batched_predictions, dim=0)

        predictions.append(batched_predictions.numpy())

        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 平均所有模型的预测（沿最后一个维度）
    mean_prediction = np.stack(predictions, axis=-1).mean(axis=-1)

    df = pd.DataFrame(mean_prediction, columns=predicted_columns)

    df = pd.concat([dm.df, df], axis=1)

    return df


def predict_using_nested_cross_validation_models(
    run_df: pd.DataFrame,
    top_k_models_to_use: int = 5,
    batch_size: int = 2048,  # 优化：从1024增加到2048
    num_workers: int = 4,
    input_df: Optional[pd.DataFrame] = None,
    use_amp: bool = False,  # 新增：是否使用混合精度
) -> pd.DataFrame:
    """
    急速版本：使用嵌套交叉验证模型进行预测

    主要优化：
    - 更大的batch size (默认2048)
    - 可选混合精度推理
    - 避免重复创建DataModule

    Args:
        run_df: MLflow runs DataFrame
        top_k_models_to_use: 每个fold使用top-k个模型
        batch_size: 批处理大小（增大以提速）
        num_workers: DataLoader工作进程数
        input_df: 输入数据DataFrame
        use_amp: 是否使用自动混合精度（FP16，约1.5-2x加速）

    Returns:
        包含预测结果的DataFrame
    """

    # Create data module (只创建一次)
    config = extract_config(run_df, run_df.run_id.iloc[0])
    config["max_utr5_len"] = 1_381  # used when training the model
    config["max_cds_utr3_len"] = 11_937  # used when training the model
    config["num_workers"] = num_workers
    config["test_batch_size"] = batch_size
    config["remove_extreme_txs"] = False
    config["target_column_pattern"] = None

    dm = RiboNNDataModule(config, df=input_df)

    # 预分配列表以避免多次扩展
    all_prediction_dfs = []

    # 获取所有unique test folds
    test_folds = np.sort(run_df["params.test_fold"].unique())

    for test_fold in test_folds:
        test_fold_str = str(test_fold)
        sub_run_df = run_df.query(
            "`params.test_fold` == @test_fold_str or `params.test_fold` == @test_fold"
        ).reset_index(drop=True)

        prediction_df = predict_using_models_trained_in_one_fold(
            sub_run_df, config, dm, top_k_models_to_use, use_amp=use_amp
        )
        prediction_df["fold"] = int(test_fold)

        all_prediction_dfs.append(prediction_df)

    # 一次性合并所有fold的预测
    all_predictions = pd.concat(all_prediction_dfs, axis=0, ignore_index=True)

    return all_predictions


# ========== 兼容性函数：保持原版API ==========

# 为了完全兼容原版，同时提供未优化版本
def predict_using_models_trained_in_one_fold_legacy(
    run_df: pd.DataFrame,
    config: Dict,
    dm,
    top_k_models_to_use: int = 5,
) -> pd.DataFrame:
    """
    原版兼容函数（调用优化版本，use_amp=False）
    """
    return predict_using_models_trained_in_one_fold(
        run_df, config, dm, top_k_models_to_use, use_amp=False
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
        run_df, top_k_models_to_use, batch_size, num_workers, input_df, use_amp=False
    )
