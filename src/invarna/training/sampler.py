"""
pretrain_data.py — self-contained MLM dataset for InvaRNA backbone pretraining.

Distilled from the original state-spaces framework (src/dataloaders/datasets/
hg38_dataset.py:mRNADataset + src/dataloaders/utils/mlm.py:mlm_getitem +
genomics.py:InvaRNADataset). Behaviour is reproduced verbatim:

  - char-level tokenize, padding="max_length", max_length, truncation, add_special_tokens=False
  - N (id 11) -> pad (id 4) before masking
  - MLM 15%: only ACGT (ids 7,8,9,10) eligible; 80% [MASK] / 10% random / 10% unchanged
  - target = pad_token_id (4) on non-masked positions (loss computed only on masked)
  - __getitem__ returns (data, target) LongTensors; default collate stacks to (B, L)
"""
import torch
import pyfastx
from torch.utils.data import Dataset


def mlm_getitem(seq, mlm_probability=0.15, contains_eos=False, tokenizer=None,
                eligible_replacements=None, eligible_mask_ids=(7, 8, 9, 10)):
    """Create (data, target) for MLM. Verbatim from src/dataloaders/utils/mlm.py."""
    data = seq[:-1].clone() if contains_eos else seq.clone()
    target = data.clone()

    probability_matrix = torch.full(target.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Restrict maskable positions (only ACGT).
    if eligible_mask_ids is not None:
        allowed_mask = torch.zeros_like(target, dtype=torch.bool)
        for token_id in eligible_mask_ids:
            allowed_mask |= (target == token_id)
        masked_indices &= allowed_mask

    target[~masked_indices] = tokenizer.pad_token_id  # loss only on masked positions

    # 80% -> [MASK]
    indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    data[indices_replaced] = tokenizer.mask_token_id

    # 10% -> random token
    indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    if eligible_replacements is not None:
        rand_choice = torch.randint(eligible_replacements.shape[0], size=target.shape)
        random_words = eligible_replacements[rand_choice]
    else:
        random_words = torch.randint(len(tokenizer), size=target.shape, dtype=torch.long)
    data[indices_random] = random_words[indices_random]

    return data, target


class MLMDataset(Dataset):
    """FASTA -> tokenized, N->pad, MLM-masked (data, target). Distilled from mRNADataset."""

    def __init__(self, fasta_file, tokenizer, max_length=10000,
                 mlm_probability=0.15, add_eos=True):
        if mlm_probability <= 0.0:
            raise ValueError(f"`mlm_probability` must be > 0.0, got {mlm_probability}.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.add_eos = add_eos
        self.fasta = pyfastx.Fasta(fasta_file, build_index=True)
        self.ids = list(self.fasta.keys())

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def replace_value(x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        entry = self.fasta[self.ids[idx]]
        seq = str(entry[:len(entry) + 1])  # ensure full read + type safety

        seq = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        if self.add_eos:
            seq.append(self.tokenizer.sep_token_id)

        seq = torch.LongTensor(seq)
        # N -> PAD (don't ask the model to predict N)
        seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int["N"], self.tokenizer.pad_token_id)

        data, target = mlm_getitem(
            seq,
            mlm_probability=self.mlm_probability,
            contains_eos=self.add_eos,
            tokenizer=self.tokenizer,
        )
        return data, target
