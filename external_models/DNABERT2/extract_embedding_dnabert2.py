import os
# Suppress TensorFlow warnings (we only use PyTorch)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict
from transformers import AutoTokenizer, AutoModel
import warnings

# Suppress transformers warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


# --- Automatic path calculation ---
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR


def _extract_single_sequence(seq, model, tokenizer, device):
    """
    Extract embeddings from a single DNA sequence.

    Args:
        seq: DNA sequence string
        model: Pre-trained BERT model
        tokenizer: Tokenizer for the model
        device: Device to run inference on (cuda or cpu)

    Returns:
        Token embeddings tensor of shape (seq_length, hidden_size)
    """
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # DNABERT2 returns (encoder_outputs, pooled_output) as a tuple
    if isinstance(outputs, tuple):
        token_embeddings = outputs[0].squeeze(0)
    else:
        token_embeddings = outputs.last_hidden_state.squeeze(0)
    return token_embeddings


def extract_dna_embeddings(
    input_data: Union[str, List[Dict[str, str]], pd.DataFrame, Dict[str, str]],
    model_dir: Path = MODEL_DIR,
    device: str = "cuda:0",
    seq_column: str = "sequence"
) -> pd.DataFrame:
    """
    Extract embeddings from DNA sequences. Supports multiple input formats.

    Args:
        input_data: Input data, supports the following formats:
            - str: Single DNA sequence string
            - Dict[str, str]: Single sequence dict with 'sequence' key (or custom seq_column)
            - List[Dict[str, str]]: List of sequence dictionaries
            - pd.DataFrame: DataFrame containing sequence column
        model_dir: Directory containing the pre-trained model
        device: Device to run inference on ('cuda:0' or 'cpu')
        seq_column: Column name for sequences in DataFrame/Dict

    Returns:
        pd.DataFrame: DataFrame with sequences and their embeddings
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        trust_remote_code=True
    )

    # Load model in float32 to avoid flash attention compatibility issues
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
        emb = _extract_single_sequence(seq, model, tokenizer, device)
        # Convert to float32 for pandas compatibility
        embeddings_list.append(emb.cpu().float())

    df["embeddings"] = embeddings_list
    df["seq_length"] = df[seq_column].apply(lambda x: len(x))

    return df


def main():
    # Example usage with different input formats

    # Scenario 1: Single sequence string
    print("--- Scenario 1: Single sequence (str) ---")
    seq_str = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    result_str = extract_dna_embeddings(seq_str)
    print(f"Sequence: {result_str.iloc[0]['sequence'][:50]}...")
    print(f"Embeddings shape: {result_str.iloc[0]['embeddings'].shape}")
    print(f"Sequence length: {result_str.iloc[0]['seq_length']}")

    # Scenario 2: Single sequence dict
    print("\n--- Scenario 2: Single sequence (Dict) ---")
    seq_dict = {"sequence": "ATGCGATCGATCGATCGAT"}
    result_dict = extract_dna_embeddings(seq_dict)
    print(f"Embeddings shape: {result_dict.iloc[0]['embeddings'].shape}")

    # Scenario 3: Multiple sequences (List[Dict])
    print("\n--- Scenario 3: Multiple sequences (List[Dict]) ---")
    seq_list = [
        {"sequence": "ATGCGATCGATCG"},
        {"sequence": "TACGATCGATCGATCGATCG"},
        {"sequence": "GCGATCGATCGATCGATCGATCG"}
    ]
    result_list = extract_dna_embeddings(seq_list)
    print(result_list[["seq_id", "seq_length"]])

    # Scenario 4: DataFrame
    print("\n--- Scenario 4: Multiple sequences (DataFrame) ---")
    df_input = pd.DataFrame({
        "sequence": ["ATGCGATCGATCG", "TACGATCGATCGATCGATCG"],
        "gene_name": ["gene1", "gene2"]
    })
    result_df = extract_dna_embeddings(df_input)
    print(result_df[["seq_id", "gene_name", "seq_length"]])


if __name__ == "__main__":
    main()
