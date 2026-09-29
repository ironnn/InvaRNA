"""
InvaRNA single-GPU inference API.

Usage:
    from invarna.evaluation.inference_api import InvaRNAPredictor

    predictor = InvaRNAPredictor("final_tdc")  # load once
    scores = predictor.predict(["ATGCCC...", "ATGAAA..."], utr5_sizes=[100, 80])
    # returns: list of floats
"""
import torch

from .human_te import (
    load_stage2, load_stage1, align_and_pad,
    STAGE2_CKPTS, STAGE1_CKPTS, TOTAL_LENGTH,
)
from invarna.models.tokenization import InvaRNATokenizer


class InvaRNAPredictor:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        """
        model_name: any key from STAGE2_CKPTS or STAGE1_CKPTS
            Use 'final_tdc' for the canonical deployed TE checkpoint. Historical
            aliases are retained only for provenance and archived-workflow checks.
        """
        self.device = torch.device(device)
        self.model_name = model_name

        if model_name in STAGE2_CKPTS:
            self.model = load_stage2(model_name, self.device)
        elif model_name in STAGE1_CKPTS:
            self.model = load_stage1(model_name, self.device)
        else:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Available: {list(STAGE2_CKPTS) + list(STAGE1_CKPTS)}")

        self.tokenizer = InvaRNATokenizer(model_max_length=TOTAL_LENGTH)
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.convert_tokens_to_ids("N")
        self.ids_to_mask = torch.tensor([self.pad_id, self.unk_id], device=self.device)

    @torch.no_grad()
    def predict(self, sequences: list, utr5_sizes: list = None, batch_size: int = 32) -> list:
        """
        sequences: list of mRNA strings
        utr5_sizes: list of int (5'UTR lengths). If None, defaults to 0 for all.
        returns: list of float predictions
        """
        if utr5_sizes is None:
            utr5_sizes = [0] * len(sequences)

        preds = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_utr5 = utr5_sizes[i:i + batch_size]

            batch_ids = []
            for seq, u5 in zip(batch_seqs, batch_utr5):
                padded = align_and_pad(seq, u5)
                ids = self.tokenizer.encode(padded, add_special_tokens=False)
                ids = [self.pad_id if t == self.unk_id else t for t in ids]
                batch_ids.append(ids)

            inp = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
            mask = (~torch.isin(inp, self.ids_to_mask)).long()

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                out = self.model(inp, attention_mask=mask)

            if hasattr(out, "logits"):
                bp = out.logits.squeeze(-1).float()
            else:
                bp = out.squeeze(-1).float()

            preds.extend([bp.item()] if bp.ndim == 0 else bp.cpu().tolist())

        return preds
