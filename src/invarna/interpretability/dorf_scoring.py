"""dORF nomination and Fig. 4 routing-prioritization helpers.

Nomination follows the recovered ``RPFDB2/01_orf_scan.py`` and
``02_rpkm_analysis.py`` implementation. Routing Z-scores follow the final
InvaRNA0412 Fig. 4B implementation.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from invarna.evaluation.inference_api import InvaRNAPredictor
from invarna.interpretability.feature_shuffle import expert_region_zscore


MIN_ORF_LENGTH = 75
MIN_MEDIAN_UTR3_RPKM = 1.0
EXPERT_NAMES = ("Single Base", "Codon", "9-mer", "Ribosome")


def score_mrnas(mrnas, utr5_sizes, model="final_tdc", device="cuda:0", batch_size=32):
    return InvaRNAPredictor(model, device=device).predict(mrnas, utr5_sizes, batch_size)


def extract_utr3(tx_sequence: str, utr5_size: int, cds_size: int) -> str:
    return str(tx_sequence)[int(utr5_size) + int(cds_size):]


def find_utr3_orfs(sequence: str, minimum_length: int = MIN_ORF_LENGTH) -> list[dict[str, Any]]:
    """Call the same ``orffinder`` implementation and options as the production scan."""
    try:
        from orffinder import orffinder
    except ImportError as exc:
        raise ImportError(
            "dORF nomination requires the `orffinder` package used by the original analysis"
        ) from exc
    if not sequence or len(sequence) < minimum_length:
        return []
    return list(orffinder.getORFs(
        SeqRecord(Seq(sequence), id="utr3"),
        minimum_length=minimum_length,
        remove_nested=True,
    ))


def summarize_orfs(orfs: list[dict[str, Any]], utr3_length: int) -> dict[str, Any]:
    covered: set[int] = set()
    for orf in orfs:
        start = min(int(orf["start"]), int(orf["end"]))
        end = max(int(orf["start"]), int(orf["end"]))
        covered.update(range(start, end + 1))
    lengths = [int(orf["length"]) for orf in orfs]
    return {
        "has_orf": bool(orfs),
        "n_orfs": len(orfs),
        "max_orf_len": max(lengths, default=0),
        "total_orf_coverage": len(covered),
        "orf_coverage_frac": len(covered) / utr3_length if utr3_length else 0.0,
    }


def nominate_dorf_transcripts(frame: pd.DataFrame, minimum_length: int = MIN_ORF_LENGTH) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Scan each mature transcript's 3' UTR and return transcript-level annotations."""
    required = {"transcript_id", "gene_id", "tx_sequence", "utr5_size", "cds_size", "utr3_size"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    records = []
    orfs_by_transcript = {}
    for row in frame.itertuples(index=False):
        utr3 = extract_utr3(row.tx_sequence, row.utr5_size, row.cds_size)
        try:
            orfs = find_utr3_orfs(utr3, minimum_length)
        except ImportError:
            raise
        except Exception:
            orfs = []
        orfs_by_transcript[row.transcript_id] = orfs
        record = {
            "transcript_id": row.transcript_id,
            "gene_id": row.gene_id,
            "SYMBOL": getattr(row, "SYMBOL", ""),
            "utr3_size": row.utr3_size,
        }
        record.update(summarize_orfs(orfs, len(utr3)))
        records.append(record)
    return pd.DataFrame(records), orfs_by_transcript


def compute_median_utr3_rpkm(count_matrix: pd.DataFrame, gene_lengths: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Production RPKM calculation: CPM divided by feature length in kb."""
    library_sizes = count_matrix.sum(axis=0)
    rpkm = count_matrix.div(library_sizes / 1e6, axis=1).div(gene_lengths / 1e3, axis=0)
    median_nonzero = rpkm.replace(0, np.nan).median(axis=1).rename("median_rpkm_3utr")
    fraction_gt1 = ((rpkm > 1).sum(axis=1) / rpkm.shape[1]).rename("frac_samples_rpkm_gt1")
    return median_nonzero, fraction_gt1


def filter_riboseq_supported_dorfs(
    annotations: pd.DataFrame,
    median_rpkm: pd.Series,
    fraction_gt1: pd.Series,
    threshold: float = MIN_MEDIAN_UTR3_RPKM,
) -> pd.DataFrame:
    """Keep ORF-positive transcripts whose nonzero-sample median 3'UTR RPKM exceeds 1."""
    table = annotations.copy()
    table["gene_base"] = table["gene_id"].astype(str).str.split(".").str[0]
    table = table.merge(median_rpkm.rename_axis("gene_base").reset_index(), on="gene_base", how="left")
    table = table.merge(fraction_gt1.rename_axis("gene_base").reset_index(), on="gene_base", how="left")
    return table[table["has_orf"] & (table["median_rpkm_3utr"] > threshold)].copy()


def expand_candidate_orfs(
    candidates: pd.DataFrame,
    transcripts: pd.DataFrame,
    orfs_by_transcript: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Create the per-ORF table used as input to genome-wide routing analysis."""
    merged = candidates.merge(
        transcripts[["transcript_id", "gene_id", "tx_sequence", "utr5_size", "cds_size"]],
        on=["transcript_id", "gene_id"], how="inner",
    )
    rows = []
    for row in merged.itertuples(index=False):
        utr3 = extract_utr3(row.tx_sequence, row.utr5_size, row.cds_size)
        for orf in orfs_by_transcript.get(row.transcript_id, []):
            start = min(int(orf["start"]), int(orf["end"]))
            end = max(int(orf["start"]), int(orf["end"])) + 1
            orf_length = int(orf["length"])
            # This is the exact recovered formula; it assumes uniform 3'UTR read density.
            orf_rpkm = row.median_rpkm_3utr * (len(utr3) / orf_length)
            rows.append({
                "transcript_id": row.transcript_id, "gene_id": row.gene_id,
                "SYMBOL": row.SYMBOL, "gene_base": row.gene_base,
                "utr5_size": row.utr5_size, "cds_size": row.cds_size,
                "utr3_size": row.utr3_size, "orf_start_in_utr3": start,
                "orf_end_in_utr3": end, "orf_len": orf_length,
                "median_rpkm_3utr": row.median_rpkm_3utr, "orf_rpkm": orf_rpkm,
                "frac_samples_rpkm_gt1": row.frac_samples_rpkm_gt1,
            })
    return pd.DataFrame(rows)


def whole_utr3_routing_zscore(routing_probabilities: np.ndarray, start: int, utr3_length: int) -> dict[str, np.ndarray]:
    """Z-score native versus whole-3'UTR shuffles over the 3'UTR routing region."""
    return expert_region_zscore(routing_probabilities, start, start + utr3_length)


def rank_dorf_candidates(frame: pd.DataFrame, expert: str = "Ribosome") -> pd.DataFrame:
    """Rank one row per symbol by descending expert Z-score, as in final Fig. 4B."""
    selected = frame[frame["Expert"] == expert].drop_duplicates("SYMBOL").copy()
    selected = selected.sort_values("Z_Score", ascending=False).reset_index(drop=True)
    selected["rank_zero_based"] = np.arange(len(selected))
    selected["top_fraction"] = 1.0 - selected["rank_zero_based"] / max(len(selected), 1)
    return selected
