import os
# Suppress TensorFlow warnings (we only use PyTorch)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict
from transformers import AutoTokenizer, BertModel


# --- Automatic path calculation ---
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR


def _extract_single_sequence(seq, model, tokenizer, device):
    """
    Extract codon embeddings from a single DNA sequence.

    Args:
        seq: DNA sequence string
        model: Pre-trained BERT model
        tokenizer: Tokenizer for the model
        device: Device to run inference on (cuda or cpu)

    Returns:
        Codon embeddings tensor of shape (1, num_codons, hidden_size)
    """
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]

    inputs = tokenizer(codons, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    codon_embeddings = outputs.last_hidden_state[:, 1:len(codons)+1, :]
    return codon_embeddings


def extract_codon_embeddings(
    input_data: Union[str, List[Dict[str, str]], pd.DataFrame, Dict[str, str]],
    model_dir: Path = MODEL_DIR,
    device: str = "cuda:0",
    seq_column: str = "sequence"
) -> pd.DataFrame:
    """
    Extract codon embeddings from DNA sequences. Supports multiple input formats.

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
        trust_remote_code=False
    )

    model = BertModel.from_pretrained(model_dir, dtype=torch.bfloat16)
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
    df["num_codons"] = df[seq_column].apply(lambda x: len(x) // 3)

    return df


def main():
    # Example usage with different input formats

    # Scenario 1: Single sequence string
    print("--- Scenario 1: Single sequence (str) ---")
    seq_str = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
    result_str = extract_codon_embeddings(seq_str)
    print(f"Sequence: {result_str.iloc[0]['sequence']}")
    print(f"Embeddings shape: {result_str.iloc[0]['embeddings'].shape}")
    print(f"Number of codons: {result_str.iloc[0]['num_codons']}")

    # Scenario 2: Single sequence dict
    print("\n--- Scenario 2: Single sequence (Dict) ---")
    seq_dict = {"sequence": "ATGAAACCCGGGTAG"}
    result_dict = extract_codon_embeddings(seq_dict)
    print(f"Embeddings shape: {result_dict.iloc[0]['embeddings'].shape}")

    # Scenario 3: Multiple sequences (List[Dict])
    print("\n--- Scenario 3: Multiple sequences (List[Dict]) ---")
    seq_list = [
        {"sequence": "ATGAAATAG"},
        {"sequence": "ATGCCCTAG"},
        {"sequence": "ATGGGGCCCAAATAG"}
    ]
    result_list = extract_codon_embeddings(seq_list)
    print(result_list[["seq_id", "num_codons"]])

    # Scenario 4: DataFrame
    print("\n--- Scenario 4: Multiple sequences (DataFrame) ---")
    df_input = pd.DataFrame({
        "sequence": ["ATGAAATAG", "ATGCCCTAG"],
        "gene_name": ["gene1", "gene2"]
    })
    result_df = extract_codon_embeddings(df_input)
    print(result_df[["seq_id", "gene_name", "num_codons"]])


if __name__ == "__main__":
    main()
