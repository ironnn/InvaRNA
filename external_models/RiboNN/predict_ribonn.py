#./predict.py

import sys
import os
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Union, Optional, Dict, List

# --- 自动路径计算 ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR / "RiboNN-1.0.0"
if str(PROJECT_ROOT) not in sys.path: sys.path.append(str(PROJECT_ROOT))

from src.predict import predict_using_nested_cross_validation_models

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(newdir)
    try: yield
    finally: os.chdir(prevdir)

def run_ribonn_prediction(
    input_data: Union[str, List[Dict[str, str]], pd.DataFrame, Dict[str, str]],
    n_folds: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    统一的 RiboNN TE 预测接口，支持多种输入格式

    Args:
        input_data: 输入数据，支持以下格式:
            - str (path): TSV 文件路径
            - Dict[str, str]: 单条序列字典，需包含 utr5_sequence, cds_sequence, utr3_sequence
            - List[Dict[str, str]]: 多条序列字典列表
            - pd.DataFrame: 包含序列列的DataFrame
        n_folds: 交叉验证折数，默认 5

    Returns:
        pd.DataFrame: 包含序列和预测结果的DataFrame
    """

    # 1. 统一转换为 DataFrame 并处理 ID
    if isinstance(input_data, dict):
        # 单条序列字典
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        # 序列字典列表
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, (str, Path)):
        # 文件路径
        df = pd.read_csv(Path(input_data).resolve(), sep="\t")
    else:
        # DataFrame
        df = input_data.copy()

    # 如果缺少 tx_id，则自动创造
    if "tx_id" not in df.columns:
        df["tx_id"] = [f"seq_{i+1}" for i in range(len(df))]

    # 2. 调用模型
    with cd(PROJECT_ROOT):
        run_df = pd.read_csv("models/runs.csv")
        run_df = run_df.head(1)
        # 严格按照位置参数传递: run_df, n_folds, batch_size, num_workers, input_df
        predictions = predict_using_nested_cross_validation_models(
            run_df, n_folds, 32, 4, df
        )

    # 3. 后处理聚合
    pred_cols = [c for c in predictions.columns if c.startswith("predicted_")]
    group_cols = [c for c in ["tx_id", "utr5_sequence", "cds_sequence", "utr3_sequence"] if c in predictions.columns]

    res = predictions.groupby(group_cols, as_index=False)[pred_cols].mean()
    res["mean_predicted_TE"] = res[pred_cols].mean(axis=1)
    return res

# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景 1: 输入单条序列 (Dict)
    print("--- 场景 1: 单条序列 (Dict) ---")
    seq_dict = {
        "utr5_sequence": "GGGAAATAAGAGAGAAAAGAAGAG",
        "cds_sequence": "ATGAAATAG",
        "utr3_sequence": "TGAGCGGCCGC"
    }
    res_single = run_ribonn_prediction(seq_dict)
    print(res_single[["tx_id", "mean_predicted_TE"]])

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
    res_list = run_ribonn_prediction(seq_list)
    print(res_list[["tx_id", "mean_predicted_TE"]])

    # 场景 3: 输入多条序列 (DataFrame)
    print("\n--- 场景 3: 多条序列 (DataFrame) ---")
    data = {
        "utr5_sequence": ["GGGAG", "AAAAA"],
        "cds_sequence": ["ATGAAATAG", "ATGCCTTAG"],
        "utr3_sequence": ["TGA", "TAG"]
    }
    # 这里故意不给 tx_id，测试自动生成功能
    df_input = pd.DataFrame(data)
    res_df = run_ribonn_prediction(df_input)
    print(res_df[["tx_id", "mean_predicted_TE"]])