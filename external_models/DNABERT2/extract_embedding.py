import os
# Suppress TensorFlow warnings (we only use PyTorch)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Literal
from transformers import AutoTokenizer, AutoModel
import logging

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


# --- Automatic path calculation ---
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = "zhihan1996/DNABERT-2-117M"  # Hugging Face model ID


def _extract_single_sequence(seq, model, tokenizer, device, pooling='none'):
    """
    Extract embeddings from a single DNA sequence.

    Args:
        seq: DNA sequence string
        model: Pre-trained BERT model
        tokenizer: Tokenizer for the model
        device: Device to run inference on (cuda or cpu)
        pooling: Pooling method - 'none', 'mean', or 'max'

    Returns:
        Token embeddings tensor
        - If pooling='none': shape (seq_length, 768)
        - If pooling='mean' or 'max': shape (768,)
    """
    inputs = tokenizer(seq, return_tensors='pt')["input_ids"]
    inputs = inputs.to(device)

    with torch.no_grad():
        hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

    # Remove batch dimension
    hidden_states = hidden_states[0]  # [sequence_length, 768]

    if pooling == 'mean':
        return torch.mean(hidden_states, dim=0)  # [768]
    elif pooling == 'max':
        return torch.max(hidden_states, dim=0)[0]  # [768]
    else:  # 'none'
        return hidden_states  # [sequence_length, 768]


def extract_dna_embeddings(
    input_data: Union[str, List[Dict[str, str]], pd.DataFrame, Dict[str, str]],
    model_dir: str = MODEL_DIR,
    device: str = "cuda:0",
    seq_column: str = "sequence",
    pooling: Literal['none', 'mean', 'max'] = 'none'
) -> pd.DataFrame:
    """
    Extract embeddings from DNA sequences. Supports multiple input formats.

    Args:
        input_data: Input data, supports the following formats:
            - str: Single DNA sequence string
            - Dict[str, str]: Single sequence dict with 'sequence' key (or custom seq_column)
            - List[Dict[str, str]]: List of sequence dictionaries
            - pd.DataFrame: DataFrame containing sequence column
        model_dir: Hugging Face model ID or local path (default: "zhihan1996/DNABERT-2-117M")
        device: Device to run inference on ('cuda:0' or 'cpu')
        seq_column: Column name for sequences in DataFrame/Dict
        pooling: Pooling method - 'none' (per-token), 'mean', or 'max'

    Returns:
        pd.DataFrame: DataFrame with sequences and their embeddings
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()

    # Convert input to DataFrame
    if isinstance(input_data, str):
        # Single sequence string
        df = pd.DataFrame([{seq_column: input_data, "seq_id": "seq_1"}])
    elif isinstance(input_data, dict):
        # Single sequence dictionary
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        # List of sequence dictionaries
        df = pd.DataFrame(input_data)
    else:
        # DataFrame
        df = input_data.copy()

    # Add seq_id if not present
    if "seq_id" not in df.columns:
        df["seq_id"] = [f"seq_{i+1}" for i in range(len(df))]

    # Extract embeddings for each sequence
    embeddings_list = []
    for _, row in df.iterrows():
        seq = row[seq_column]
        emb = _extract_single_sequence(seq, model, tokenizer, device, pooling)
        embeddings_list.append(emb.cpu().float())

    df["embeddings"] = embeddings_list
    df["seq_length"] = df[seq_column].apply(lambda x: len(x))

    if pooling != 'none':
        df["embedding_dim"] = 768

    return df


def main():
    # Example usage with different input formats and pooling methods

    # Scenario 1: Single sequence string with no pooling (per-token embeddings)
    print("--- Scenario 1: Single sequence with per-token embeddings ---")
    seq_str = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    result_str = extract_dna_embeddings(seq_str, pooling='none')
    print(f"Sequence length: {result_str.iloc[0]['seq_length']}")
    print(f"Embeddings shape: {result_str.iloc[0]['embeddings'].shape}")

    # Scenario 2: Single sequence with mean pooling
    print("\n--- Scenario 2: Single sequence with mean pooling ---")
    result_mean = extract_dna_embeddings(seq_str, pooling='mean')
    print(f"Embeddings shape: {result_mean.iloc[0]['embeddings'].shape}")
    print(f"Embedding dimension: {result_mean.iloc[0]['embedding_dim']}")

    # Scenario 3: Single sequence with max pooling
    print("\n--- Scenario 3: Single sequence with max pooling ---")
    result_max = extract_dna_embeddings(seq_str, pooling='max')
    print(f"Embeddings shape: {result_max.iloc[0]['embeddings'].shape}")

    # Scenario 4: Multiple sequences (List[Dict]) with mean pooling
    print("\n--- Scenario 4: Multiple sequences with mean pooling ---")
    seq_list = [
        {"sequence": "ATGCGATCGATCG"},
        {"sequence": "TACGATCGATCGATCGATCG"},
        {"sequence": "GCGATCGATCGATCGATCGATCG"}
    ]
    result_list = extract_dna_embeddings(seq_list, pooling='mean')
    print(result_list[["seq_id", "seq_length", "embedding_dim"]])

    # Scenario 5: DataFrame with metadata
    print("\n--- Scenario 5: DataFrame with metadata ---")
    df_input = pd.DataFrame({
        "sequence": ["ATGCGATCGATCG", "TACGATCGATCGATCGATCG"],
        "gene_name": ["gene1", "gene2"]
    })
    result_df = extract_dna_embeddings(df_input, pooling='mean')
    print(result_df[["seq_id", "gene_name", "seq_length", "embedding_dim"]])


if __name__ == "__main__":
    main()
