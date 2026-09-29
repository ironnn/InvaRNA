"""
UTR-LM TE 推理脚本
直接从 model_architecture.py 导入 MJ4 训练脚本的模型结构
只加载 HEK 细胞系的模型权重进行推理
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Union, List, Dict
from pathlib import Path

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'Scripts'))
sys.path.insert(0, BASE_DIR)  # 添加当前目录到路径

from esm.data import Alphabet

# 从独立模块导入模型架构（与 MJ4 训练脚本完全一致）
from model_architecture import CNN_linear

# 配置参数（与 MJ4 训练脚本一致）
LAYERS = 6
HEADS = 16
EMBED_DIM = 128
INP_LEN = 100
NODES = 40
DROPOUT3 = 0.2
CNN_LAYERS = 0
AVG_EMB = False
BOS_EMB = True
MAGIC = False
MODELFILE = 'ESM2SI_3.1'

# HEK 模型权重路径（10个fold）
HEK_MODEL_PATHS = [
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold0_epoch275.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold1_epoch295.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold2_epoch149.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold3_epoch223.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold4_epoch280.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold5_epoch101.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold6_epoch281.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold7_epoch210.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold8_epoch268.pt"),
    os.path.join(BASE_DIR, "Model/Downstream/TE_EL/MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold9_epoch280.pt"),
]


class TEPredictor:
    """单个模型推理器"""

    def __init__(self, model_path: str, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.alphabet = Alphabet(standard_toks='AGCT', mask_prob=0.0)

        # 使用从 MJ4 提取的模型结构，参数与训练脚本完全一致
        self.model = CNN_linear(
            layers=LAYERS,
            heads=HEADS,
            embed_dim=EMBED_DIM,
            inp_len=INP_LEN,
            nodes=NODES,
            dropout3=DROPOUT3,
            cnn_layers=CNN_LAYERS,
            avg_emb=AVG_EMB,
            bos_emb=BOS_EMB,
            magic=MAGIC,
            modelfile=MODELFILE
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        # 移除 'module.' 前缀（如果是 DDP 训练的）
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # print(f"[UTR-LM] Loaded model from: {os.path.basename(model_path)}")

    def _tokenize(self, seq: str) -> torch.Tensor:
        """序列tokenize"""
        seq = seq.upper().replace('U', 'T')[-INP_LEN:]  # 取后100个碱基
        tokens = [self.alphabet.cls_idx]
        tokens.extend([self.alphabet.tok_to_idx.get(c, self.alphabet.unk_idx) for c in seq])
        tokens.append(self.alphabet.eos_idx)
        return torch.tensor(tokens, dtype=torch.long)

    def predict(self, sequence: str) -> float:
        """预测单个序列的 TE 值"""
        tokens = self._tokenize(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tokens)
        return output.item()

    def predict_batch(self, sequences: List[str]) -> List[float]:
        """批量预测"""
        tokens_list = [self._tokenize(seq) for seq in sequences]
        max_len = max(len(t) for t in tokens_list)

        # Padding
        padded = []
        for t in tokens_list:
            if len(t) < max_len:
                t = torch.cat([
                    t,
                    torch.full((max_len - len(t),), self.alphabet.padding_idx, dtype=torch.long)
                ])
            padded.append(t)

        batch = torch.stack(padded).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)

        return outputs.squeeze(-1).tolist()


class TEEnsemblePredictor:
    """集成模型推理器（10个fold模型）"""

    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictors = []

        print(f"[UTR-LM] Loading HEK ensemble models (10 folds)...")
        for i, model_path in enumerate(HEK_MODEL_PATHS):
            try:
                predictor = TEPredictor(model_path, device=self.device)
                self.predictors.append(predictor)
            except FileNotFoundError as e:
                print(f"[Warning] Fold {i} model not found: {model_path}")
                continue

        if not self.predictors:
            raise RuntimeError("No models loaded successfully!")

        print(f"[UTR-LM] Successfully loaded {len(self.predictors)}/10 models")

    def predict(self, sequence: str) -> float:
        """集成预测单个序列"""
        predictions = [p.predict(sequence) for p in self.predictors]
        return float(np.mean(predictions))

    def predict_batch(self, sequences: List[str]) -> List[float]:
        """集成预测批量序列"""
        all_predictions = np.array([p.predict_batch(sequences) for p in self.predictors])
        # 对每个序列，取所有模型的平均值
        return np.mean(all_predictions, axis=0).tolist()

    def predict_with_uncertainty(self, sequence: str) -> tuple:
        """预测并返回不确定性（标准差）"""
        predictions = [p.predict(sequence) for p in self.predictors]
        return float(np.mean(predictions)), float(np.std(predictions))


# 全局 ensemble cache（与 predict_te.py 逻辑一致）
_ensemble_cache = {}


def predict_te_ensemble(sequence: str, cell_line: str = 'HEK') -> float:
    """
    预测 5' UTR 序列的翻译效率 (TE)
    使用 10 fold 集成模型，预测完取 mean

    Args:
        sequence: 5' UTR 序列 (DNA: AGCT 或 RNA: AGCU)
        cell_line: 细胞系，目前只支持 'HEK'

    Returns:
        float: 预测的 TE 值 (log scale)
    """
    if cell_line != 'HEK':
        raise ValueError(f"Currently only supports 'HEK' cell line, got '{cell_line}'")

    # 如果 cache 中没有，加载所有 10 个模型
    if cell_line not in _ensemble_cache:
        predictors = []
        print(f"[UTR-LM] Loading {cell_line} ensemble models (10 folds)...")
        for i, model_path in enumerate(HEK_MODEL_PATHS):
            try:
                predictor = TEPredictor(model_path)
                predictors.append(predictor)
            except Exception as e:
                print(f"[Warning] Fold {i} model load failed: {e}")
                continue

        if not predictors:
            raise RuntimeError("No models loaded successfully!")

        _ensemble_cache[cell_line] = predictors
        print(f"[UTR-LM] Successfully loaded {len(predictors)}/10 models")

    # 获取缓存的 predictors
    predictors = _ensemble_cache[cell_line]

    # 所有模型预测，取 mean
    preds = [predictor.predict(sequence) for predictor in predictors]
    return float(np.mean(preds))


def predict_te(sequence: str, cell_line: str = 'HEK') -> float:
    """
    预测 5' UTR 序列的翻译效率 (TE)
    默认使用集成模型（10 fold 取 mean）

    这是主推理接口，与 predict_te.py 的 predict_te 函数逻辑一致

    Args:
        sequence: 5' UTR 序列 (DNA: AGCT 或 RNA: AGCU)
        cell_line: 细胞系，默认 'HEK'

    Returns:
        float: 预测的 TE 值 (log scale)
    """
    return predict_te_ensemble(sequence, cell_line=cell_line)


def predict_te_batch(sequences: List[str], cell_line: str = 'HEK') -> List[float]:
    """
    批量预测 5' UTR 序列的翻译效率 (TE)
    使用 10 fold 集成模型

    Args:
        sequences: 5' UTR 序列列表
        cell_line: 细胞系，默认 'HEK'

    Returns:
        List[float]: 预测的 TE 值列表 (log scale)
    """
    if cell_line != 'HEK':
        raise ValueError(f"Currently only supports 'HEK' cell line, got '{cell_line}'")

    # 如果 cache 中没有，加载所有 10 个模型
    if cell_line not in _ensemble_cache:
        predictors = []
        print(f"[UTR-LM] Loading {cell_line} ensemble models (10 folds)...")
        for i, model_path in enumerate(HEK_MODEL_PATHS):
            try:
                predictor = TEPredictor(model_path)
                predictors.append(predictor)
            except Exception as e:
                print(f"[Warning] Fold {i} model load failed: {e}")
                continue

        if not predictors:
            raise RuntimeError("No models loaded successfully!")

        _ensemble_cache[cell_line] = predictors
        print(f"[UTR-LM] Successfully loaded {len(predictors)}/10 models")

    # 获取缓存的 predictors
    predictors = _ensemble_cache[cell_line]

    # 所有模型批量预测，取 mean
    all_predictions = np.array([predictor.predict_batch(sequences) for predictor in predictors])
    return np.mean(all_predictions, axis=0).tolist()


def run_utrlm_prediction(
    input_data: Union[str, List[str], List[Dict[str, str]], pd.DataFrame, Dict[str, str]],
    cell_line: str = 'HEK',
    sequence_column: str = 'utr5_sequence',
    **kwargs
) -> pd.DataFrame:
    """
    统一的 UTR-LM TE 预测接口，支持多种输入格式

    与 RiboNN 接口保持一致，支持相同的输入格式（包含 utr5/cds/utr3），
    但 UTR-LM 只使用 utr5_sequence 进行预测

    Args:
        input_data: 输入数据，支持以下格式:
            - str: 单条序列字符串 或 文件路径
            - List[str]: 序列列表
            - Dict[str, str]: 序列字典，可包含 utr5_sequence, cds_sequence, utr3_sequence
            - List[Dict[str, str]]: 多条序列字典列表
            - pd.DataFrame: 包含序列列的DataFrame
        cell_line: 细胞系，默认 'HEK'
        sequence_column: 使用的序列列名，默认 'utr5_sequence'

    Returns:
        pd.DataFrame: 包含输入序列和预测结果的DataFrame
    """

    # 1. 统一转换为 DataFrame 格式
    if isinstance(input_data, str):
        # 判断是文件路径还是序列
        input_path = Path(input_data)
        if input_path.exists() and input_path.is_file():
            # 是文件路径
            if str(input_path).endswith('.csv'):
                df = pd.read_csv(input_path)
            else:
                df = pd.read_csv(input_path, sep="\t")
        else:
            # 是单条序列字符串
            df = pd.DataFrame([{sequence_column: input_data}])
    elif isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Input list is empty")
        # 判断是字符串列表还是字典列表
        if isinstance(input_data[0], str):
            # 序列字符串列表
            df = pd.DataFrame([{sequence_column: seq} for seq in input_data])
        elif isinstance(input_data[0], dict):
            # 序列字典列表（与 RiboNN 格式一致）
            df = pd.DataFrame(input_data)
        else:
            raise ValueError(f"Unsupported list element type: {type(input_data[0])}")
    elif isinstance(input_data, dict):
        # 字典格式（与 RiboNN 格式一致）
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")

    # 2. 检查必需的列
    if sequence_column not in df.columns:
        raise ValueError(f"Column '{sequence_column}' not found in input data. Available columns: {df.columns.tolist()}")

    # 3. 如果缺少 ID 列，自动创建
    id_column = 'seq_id'
    if id_column not in df.columns:
        df[id_column] = [f"seq_{i+1}" for i in range(len(df))]

    # 4. 批量预测（只使用 utr5_sequence）
    sequences = df[sequence_column].tolist()
    predictions = predict_te_batch(sequences, cell_line=cell_line)

    # 5. 添加预测结果到 DataFrame
    df['predicted_TE'] = predictions

    return df


if __name__ == '__main__':
    # ================= 使用示例 =================

    # 场景 1: 输入单条序列 (字符串)
    print("--- 场景 1: 单条序列 (字符串) ---")
    single_seq = "AGCTGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"
    res_single = run_utrlm_prediction(single_seq)
    print(res_single[["seq_id", "predicted_TE"]])

    # 场景 2: 输入单条序列 (Dict - 与 RiboNN 格式一致)
    print("\n--- 场景 2: 单条序列 (Dict - 含 utr5/cds/utr3) ---")
    seq_dict = {
        "utr5_sequence": "GGGAAATAAGAGAGAAAAGAAGAG",
        "cds_sequence": "ATGAAATAG",
        "utr3_sequence": "TGAGCGGCCGC"
    }
    res_dict = run_utrlm_prediction(seq_dict)
    print(res_dict[["seq_id", "utr5_sequence", "predicted_TE"]])
    print("注意: UTR-LM 只使用 utr5_sequence 进行预测")

    # 场景 3: 输入多条序列 (List[str])
    print("\n--- 场景 3: 多条序列 (List[str]) ---")
    seq_list = [
        "AGCTGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT",
        "ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT",
    ]
    res_list = run_utrlm_prediction(seq_list)
    print(res_list[["seq_id", "predicted_TE"]])

    # 场景 4: 输入多条序列 (List[Dict] - 与 RiboNN 格式一致)
    print("\n--- 场景 4: 多条序列 (List[Dict] - 含 utr5/cds/utr3) ---")
    seq_dict_list = [
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
    res_dict_list = run_utrlm_prediction(seq_dict_list)
    print(res_dict_list[["seq_id", "utr5_sequence", "predicted_TE"]])

    # 场景 5: 输入多条序列 (DataFrame - 与 RiboNN 格式一致)
    print("\n--- 场景 5: 多条序列 (DataFrame - 含 utr5/cds/utr3) ---")
    import pandas as pd
    data = {
        "utr5_sequence": ["GGGAG", "AAAAA"],
        "cds_sequence": ["ATGAAATAG", "ATGCCTTAG"],
        "utr3_sequence": ["TGA", "TAG"]
    }
    df_input = pd.DataFrame(data)
    res_df = run_utrlm_prediction(df_input)
    print(res_df[["seq_id", "utr5_sequence", "predicted_TE"]])


