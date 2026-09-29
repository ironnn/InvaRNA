"""
统一 TE 推理调度器
==================
对 llm_models/inference/ 下的第三方 TE 预测模型提供统一调用入口。
改编自 agent2/predict_all_models.py, 去掉了 Ouroboros (老版 InvaRNA)。

支持的模型 (models=[...]):
  ribonn   RiboNN   (utr5+cds+utr3, 需要 conda 环境 'mamballm')   inference/ribonn
  utrlm    UTR-LM   (只用 utr5_sequence, 10-fold ensemble)        inference/UTR_LM

输入支持: 文件路径(csv/tsv/parquet) | dict | list[dict] | pd.DataFrame
需要的列: utr5_sequence, cds_sequence, utr3_sequence
          或  mrna + utr5_size (+ cds_size 用于拆分)

输出: 原始列 + RiboNN_TE, UTRLM_TE

命令行用法 (跑内置示例):
  python run_inference.py
编程用法:
  from run_inference import predict_with_all_models
  df = predict_with_all_models({"utr5_sequence":"GGGAG","cds_sequence":"ATGAAATAG","utr3_sequence":"TGA"})
"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import Union, List, Dict

BASE_DIR = Path(__file__).parent.resolve()
LLM_ROOT = BASE_DIR.parent          # llm_models/
INF_DIR = LLM_ROOT / "models"       # ribonn / UTR_LM now live under models/

# UTR-LM 的基础路径 (esm 模块 + 模型架构)
_basic_paths = [
    str(INF_DIR / "UTR_LM" / "Scripts"),
    str(INF_DIR / "UTR_LM"),
]
for _p in _basic_paths:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# 延迟导入缓存
_ribonn_module = None
_utrlm_module = None


def _get_ribonn_prediction_func():
    """延迟导入仓库内复制的 RiboNN 预测函数。"""
    global _ribonn_module
    if _ribonn_module is None:
        ribonn_paths = [
            str(INF_DIR / "ribonn" / "RiboNN-1.0.0"),  # src 模块
            str(INF_DIR / "ribonn"),                   # predict_ribonn 模块
        ]
        for path in reversed(ribonn_paths):
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)

        import src           # noqa: F401  RiboNN 的 src 包
        import src.predict   # noqa: F401
        import predict_ribonn
        _ribonn_module = predict_ribonn

    return _ribonn_module.run_ribonn_prediction


def _get_utrlm_prediction_func():
    """延迟导入 UTR-LM 预测函数。"""
    global _utrlm_module
    if _utrlm_module is None:
        import predict_utrlm
        _utrlm_module = predict_utrlm
    return _utrlm_module.run_utrlm_prediction


def predict_with_all_models(
    input_data: Union[Dict[str, str], List[Dict[str, str]], pd.DataFrame, str],
    models: List[str] = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    使用 RiboNN + UTR-LM 进行 TE 预测。

    Args:
        input_data: 文件路径(csv/tsv/parquet) | dict | list[dict] | DataFrame
            字段需含 utr5_sequence/cds_sequence/utr3_sequence, 或 mrna+utr5_size
        models: 要使用的模型, 子集 of ['ribonn', 'utrlm'], 默认全用
        verbose: 是否打印进度

    Returns:
        pd.DataFrame: 原始列 + RiboNN_TE + UTRLM_TE
    """
    if models is None:
        models = ["ribonn", "utrlm"]

    # 统一转换为 DataFrame
    if isinstance(input_data, str):
        input_path = Path(input_data)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_data}")
        if str(input_path).endswith(".parquet"):
            df = pd.read_parquet(input_path)
        elif str(input_path).endswith(".csv"):
            df = pd.read_csv(input_path)
        else:
            df = pd.read_csv(input_path, sep="\t")
    elif isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")

    _prepare_data(df)
    df["_temp_index"] = range(len(df))

    if verbose:
        print("=" * 70)
        print("Multi-Model TE Prediction (RiboNN + UTR-LM)")
        print("=" * 70)
        print(f"Input sequences: {len(df)}")
        print(f"Models to use: {', '.join(models)}")
        print()

    results = {}
    total_models = len(models)
    current_step = 0

    # 1. RiboNN
    if "ribonn" in models:
        current_step += 1
        if verbose:
            print(f"[{current_step}/{total_models}] Running RiboNN prediction...")
        try:
            run_ribonn_prediction = _get_ribonn_prediction_func()
            ribonn_result = run_ribonn_prediction(df)
            merge_cols = ["utr5_sequence", "cds_sequence", "utr3_sequence"]
            ribonn_result = ribonn_result[merge_cols + ["mean_predicted_TE"]].copy()
            results["ribonn_result"] = ribonn_result
            if verbose:
                n_predicted = len(ribonn_result)
                n_filtered = len(df) - n_predicted
                if n_filtered > 0:
                    print(f"   {n_filtered} sequences filtered by RiboNN")
                print(f"RiboNN completed ({n_predicted}/{len(df)} sequences)")
        except Exception as e:
            if verbose:
                print(f"RiboNN failed: {e}")
            results["ribonn_result"] = None

    # 2. UTR-LM
    if "utrlm" in models:
        current_step += 1
        if verbose:
            print(f"[{current_step}/{total_models}] Running UTR-LM prediction...")
        try:
            run_utrlm_prediction = _get_utrlm_prediction_func()
            utrlm_result = run_utrlm_prediction(df)
            results["utrlm_result"] = utrlm_result[["_temp_index", "predicted_TE"]].copy()
            if verbose:
                print("UTR-LM completed")
        except Exception as e:
            if verbose:
                print(f"UTR-LM failed: {e}")
            results["utrlm_result"] = None

    # 合并结果
    result_df = df.copy()

    if results.get("ribonn_result") is not None:
        ribonn_data = results["ribonn_result"]
        result_df = result_df.merge(
            ribonn_data,
            on=["utr5_sequence", "cds_sequence", "utr3_sequence"],
            how="left",
        )
        result_df.rename(columns={"mean_predicted_TE": "RiboNN_TE"}, inplace=True)
    elif "ribonn" in models:
        result_df["RiboNN_TE"] = None

    if results.get("utrlm_result") is not None:
        utrlm_data = results["utrlm_result"]
        result_df = result_df.merge(utrlm_data, on="_temp_index", how="left")
        result_df.rename(columns={"predicted_TE": "UTRLM_TE"}, inplace=True)
    elif "utrlm" in models:
        result_df["UTRLM_TE"] = None

    result_df.drop(columns=["_temp_index"], inplace=True)

    if verbose:
        print()
        print("=" * 70)
        print("All predictions completed!")
        print("=" * 70)

    return result_df


def _prepare_data(df: pd.DataFrame):
    """验证/补全数据列, 自动在 split-seq 与 mrna 之间换算。"""
    has_split_seq = all(c in df.columns for c in ["utr5_sequence", "cds_sequence", "utr3_sequence"])
    has_mrna = "mrna" in df.columns and "utr5_size" in df.columns

    if not has_split_seq and not has_mrna:
        raise ValueError(
            "Input data must contain either:\n"
            "  1. utr5_sequence, cds_sequence, utr3_sequence\n"
            "  OR\n"
            "  2. mrna and utr5_size"
        )

    if has_split_seq and not has_mrna:
        df["mrna"] = df["utr5_sequence"] + df["cds_sequence"] + df["utr3_sequence"]
        df["utr5_size"] = df["utr5_sequence"].str.len()
    elif has_mrna and not has_split_seq:
        has_sizes = all(c in df.columns for c in ["utr5_size", "cds_size", "utr3_size"])
        if has_sizes:
            df["utr5_sequence"] = df.apply(lambda r: r["mrna"][: r["utr5_size"]], axis=1)
            df["cds_sequence"] = df.apply(
                lambda r: r["mrna"][r["utr5_size"]: r["utr5_size"] + r["cds_size"]], axis=1)
            df["utr3_sequence"] = df.apply(
                lambda r: r["mrna"][r["utr5_size"] + r["cds_size"]:], axis=1)
        else:
            raise ValueError(
                "Cannot split mrna sequence. Provide utr5/cds/utr3_sequence "
                "or utr5_size/cds_size/utr3_size."
            )

    if "seq_id" not in df.columns:
        df["seq_id"] = [f"seq_{i+1}" for i in range(len(df))]


if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Model TE Prediction - Usage Examples")
    print("=" * 70)

    single_seq = {
        "utr5_sequence": "GGGAAATAAGAGAGAAAAGAAGAG",
        "cds_sequence": "ATGAAATAG",
        "utr3_sequence": "TGAGCGGCCGC",
    }
    result = predict_with_all_models(single_seq, verbose=True)
    print("\nResults:")
    print(result[["seq_id", "RiboNN_TE", "UTRLM_TE"]])
