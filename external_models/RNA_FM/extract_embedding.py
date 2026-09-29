import os
import torch
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Literal

from multimolecule import RnaTokenizer, RnaFmModel

import warnings
warnings.filterwarnings("ignore")

# --- Automatic path calculation ---
# Weights (model.safetensors / pytorch_model.bin + config/tokenizer) live in this dir.
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR


def _extract_single_sequence(seq, model, tokenizer, device, pooling="none"):
    """
    Extract embeddings from a single RNA sequence.

    Returns:
        - pooling='none': shape (seq_length, hidden_size)
        - pooling='mean'/'max': shape (hidden_size,)
    """
    inputs = tokenizer([seq], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state.squeeze(0)  # (seq_length, hidden_size)

    if pooling == "mean":
        return torch.mean(hidden_states, dim=0)
    elif pooling == "max":
        return torch.max(hidden_states, dim=0)[0]
    else:  # 'none'
        return hidden_states


def extract_rna_embeddings(
    input_data: Union[str, List[Dict[str, str]], pd.DataFrame, Dict[str, str]],
    model_dir: str = str(MODEL_DIR),
    device: str = "cuda:0",
    seq_column: str = "sequence",
    pooling: Literal["none", "mean", "max"] = "none",
) -> pd.DataFrame:
    """
    Extract RNA embeddings using RNA-FM (multimolecule). Supports multiple input formats.

    Args:
        input_data: str | Dict | List[Dict] | DataFrame
        model_dir: local path to the RNA-FM weights (default: this script's dir)
        device: 'cuda:0' or 'cpu'
        seq_column: sequence column name for DataFrame/Dict
        pooling: 'none' (per-token), 'mean', or 'max'

    Returns:
        pd.DataFrame with 'embeddings' and 'seq_length' columns.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RnaTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = RnaFmModel.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()

    # Convert input to DataFrame
    if isinstance(input_data, str):
        df = pd.DataFrame([{seq_column: input_data, "seq_id": "seq_1"}])
    elif isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()

    if "seq_id" not in df.columns:
        df["seq_id"] = [f"seq_{i+1}" for i in range(len(df))]

    embeddings_list = []
    for _, row in df.iterrows():
        seq = row[seq_column]
        emb = _extract_single_sequence(seq, model, tokenizer, device, pooling)
        embeddings_list.append(emb.cpu().float())

    df["embeddings"] = embeddings_list
    df["seq_length"] = df[seq_column].apply(lambda x: len(x))

    if pooling != "none":
        df["embedding_dim"] = int(embeddings_list[0].shape[-1]) if embeddings_list else None

    return df


def main():
    print("--- Scenario 1: Single sequence with per-token embeddings ---")
    seq_str = "AUGGCUACGUAUCGAUCG"
    result_str = extract_rna_embeddings(seq_str, pooling="none")
    print(f"Sequence length: {result_str.iloc[0]['seq_length']}")
    print(f"Embeddings shape: {result_str.iloc[0]['embeddings'].shape}")

    print("\n--- Scenario 2: Single sequence with mean pooling ---")
    result_mean = extract_rna_embeddings(seq_str, pooling="mean")
    print(f"Embeddings shape: {result_mean.iloc[0]['embeddings'].shape}")

    print("\n--- Scenario 3: Multiple sequences with mean pooling ---")
    seq_list = [
        {"sequence": "AUGGCUACGUAUCGAUCG"},
        {"sequence": "GGUCCUCUCUGGUUAGACCAGAUCUGAGCCU"},
    ]
    result_list = extract_rna_embeddings(seq_list, pooling="mean")
    print(result_list[["seq_id", "seq_length"]])


if __name__ == "__main__":
    main()
