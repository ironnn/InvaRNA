import re
import math
import torch
from typing import Callable, Tuple, List, Dict, Set
from encoding import encode_seq, decode_seq

from Bio.Seq import Seq
from Bio.Data import CodonTable

BASES = ["A", "C", "G", "T"]
BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX2BASE = {0: "A", 1: "C", 2: "G", 3: "T"}

# ---- Safety motifs (checked locally around mutation site) ----
FORBIDDEN_ENZYMES = [
    "GAATTC", "GGATCC", "AAGCTT", "GCGGCCGC", "CTCGAG", "GCTAGC",
]
MIRNA_SEEDS = [
    "CACTCCA", "TGCCAAA", "ACTGACA", "GTTCTCA", "AGCATTA", "ACACTAC",
    "TGCCTTA", "TCGATAC", "ACTGTGA", "CTACCTC", "TGCTGCT", "TAAGCTA",
    "CGCACAG",
]
ARE_MOTIFS = ["ATTTATTTA", "TATTTAT"]
G4_PATTERN = re.compile(r"([G]{3,}\w{1,7}){3,}[G]{3,}")

REJECTION_PENALTY = -0.05
EDIT_CONSTRAINT_PENALTY = -10.0
LOCAL_WINDOW = 15  # check ±15nt around mutation site


def _has_bad_motif(seq_fragment: str) -> bool:
    """Check if a local sequence fragment contains any forbidden motif."""
    for m in FORBIDDEN_ENZYMES:
        if m in seq_fragment:
            return True
    for m in MIRNA_SEEDS:
        if m in seq_fragment:
            return True
    for m in ARE_MOTIFS:
        if m in seq_fragment:
            return True
    if G4_PATTERN.search(seq_fragment):
        return True
    return False


def precompute_codon_action_table(cds_seq: str):
    """
    Precompute synonymous codon replacements for each codon position.
    Returns: codon_actions, n_codons, action_map
    """
    cds_seq = cds_seq.upper().replace("U", "T")
    L = len(cds_seq)
    if L % 3 != 0:
        cds_seq = cds_seq[:L - (L % 3)]

    table = CodonTable.unambiguous_dna_by_id[1]
    all_bases = "ACGT"
    all_codons = [a + b + c for a in all_bases for b in all_bases for c in all_bases]
    aa_to_codons: Dict[str, Set[str]] = {}
    for codon in all_codons:
        try:
            aa = str(Seq(codon).translate(table=table))
            aa_to_codons.setdefault(aa, set()).add(codon)
        except Exception:
            pass

    n_codons = len(cds_seq) // 3
    codon_actions = []
    action_map = []

    for i in range(n_codons):
        wt_codon = cds_seq[i * 3: i * 3 + 3]
        aa = str(Seq(wt_codon).translate(table=table))

        if aa == "*":
            codon_actions.append([])
            continue

        syn = sorted(c for c in aa_to_codons.get(aa, {wt_codon}) if c != wt_codon)
        codon_actions.append(syn)

        for new_codon in syn:
            action_map.append((i, new_codon))

    return codon_actions, n_codons, action_map


class SequenceEnv:
    """
    Codon-level Sequence Optimization Environment.

    Safety check: after each codon replacement, check ±15nt around the mutation site
    for forbidden motifs (enzymes, G4, miRNA seeds, ARE). If found, the mutation is
    rejected (rolled back), the step is consumed, and reward = -0.05.
    """

    def __init__(
        self,
        init_seq: str,
        genename: str,
        editregion: str,
        max_steps: int,
        reward_fn: Callable[[str, str, str], Tuple[float, Dict[str, float]]],
        reward_scale: float = 1.0,
        reward_mode: str = "terminal",
        device: str = "cpu",
        **kwargs
    ):
        self.init_seq_str = init_seq.upper().replace("U", "T")
        self.genename = genename
        self.editregion = editregion.lower()
        self.L = len(self.init_seq_str)
        self.max_steps = max_steps
        self.device = device

        self.reward_fn = reward_fn
        self.reward_scale = reward_scale
        self.reward_mode = reward_mode
        self.min_edit_fraction = float(kwargs.get("min_edit_fraction", 0.0))
        if not 0.0 <= self.min_edit_fraction <= 1.0:
            raise ValueError("min_edit_fraction must be between 0 and 1")
        self.min_edit_count = math.ceil(self.L * self.min_edit_fraction)

        if self.editregion == "cds":
            self.codon_actions, self.n_codons, self.action_map = \
                precompute_codon_action_table(self.init_seq_str)
            self.action_dim = len(self.action_map)
            self.is_codon_mode = True
            print(f"[Env] CDS codon-level actions: {self.action_dim} synonymous replacements "
                  f"across {self.n_codons} codons")
        else:
            self.action_dim = self.L * 4
            self.is_codon_mode = False
            self.action_map = None
            print(f"[Env] {self.editregion.upper()} base-level actions: {self.action_dim}")

        self.reset()

    def reset(self):
        self.seq = encode_seq(self.init_seq_str).clone().detach().to(self.device)
        self.t = 0
        self.done = False
        self.score_cache = {}
        self.rejected_count = 0
        return self.observe()

    def observe(self):
        return self.seq.detach().clone()

    def _seq_str(self) -> str:
        """Current sequence as string."""
        return decode_seq(self.seq)

    def _check_local_safety(self, codon_idx: int, new_codon: str) -> bool:
        """
        Check ±15nt around mutation site for bad motifs.
        Only converts the local window to string (not the full sequence).
        """
        start = codon_idx * 3
        window_start = max(0, start - LOCAL_WINDOW)
        window_end = min(self.L, start + 3 + LOCAL_WINDOW)

        # Convert only the local window to string
        local = []
        for i in range(window_start, window_end):
            if start <= i < start + 3:
                local.append(new_codon[i - start])
            else:
                local.append(IDX2BASE[self.seq[i].item()])
        fragment = "".join(local)

        return not _has_bad_motif(fragment)

    def _check_local_base_safety(self, pos: int, new_base: str) -> bool:
        """Check the same local safety motifs for a proposed UTR base edit."""
        window_start = max(0, pos - LOCAL_WINDOW)
        window_end = min(self.L, pos + 1 + LOCAL_WINDOW)
        local = []
        for i in range(window_start, window_end):
            local.append(new_base if i == pos else IDX2BASE[self.seq[i].item()])
        return not _has_bad_motif("".join(local))

    def _mutation_stats(self):
        current = self._seq_str()
        count = sum(a != b for a, b in zip(current, self.init_seq_str))
        fraction = count / self.L if self.L else 0.0
        return count, fraction

    def _score(self, seq_tensor: torch.LongTensor) -> Tuple[float, Dict[str, float]]:
        seq_region = decode_seq(seq_tensor)
        if seq_region in self.score_cache:
            return self.score_cache[seq_region]
        try:
            raw_reward, metrics = self.reward_fn(seq_region, self.genename, self.editregion)
        except Exception as e:
            print(f"Error in reward calculation: {e}")
            raw_reward = 0.0
            metrics = {}
        final_reward = raw_reward * self.reward_scale
        self.score_cache[seq_region] = (final_reward, metrics)
        return final_reward, metrics

    def step(self, action_id: int):
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        rejected = False

        if self.is_codon_mode:
            codon_idx, new_codon = self.action_map[action_id]

            # Safety check: ±15nt local window
            if self._check_local_safety(codon_idx, new_codon):
                # Safe — apply mutation
                start = codon_idx * 3
                for j, base in enumerate(new_codon):
                    self.seq[start + j] = BASE2IDX[base]
            else:
                # Rejected — don't mutate, consume step
                rejected = True
                self.rejected_count += 1
        else:
            # UTR base-level mutation with the same local motif safety filter.
            pos = action_id // 4
            base_idx = action_id % 4
            new_base = IDX2BASE[base_idx]
            if self._check_local_base_safety(pos, new_base):
                self.seq[pos] = base_idx
            else:
                rejected = True
                self.rejected_count += 1

        self.t += 1
        self.done = self.t >= self.max_steps

        reward = 0.0
        metrics = {}

        if self.done and self.reward_mode == "terminal":
            mutation_count, mutation_fraction = self._mutation_stats()
            constraint_valid = mutation_count >= self.min_edit_count
            if constraint_valid:
                reward, metrics = self._score(self.seq)
            else:
                reward = EDIT_CONSTRAINT_PENALTY
                metrics = {}
            metrics.update({
                "mutation_count": mutation_count,
                "mutation_fraction": mutation_fraction,
                "min_edit_count": self.min_edit_count,
                "edit_constraint_valid": float(constraint_valid),
                "rejected": float(rejected),
            })
            if rejected and constraint_valid:
                reward += REJECTION_PENALTY
        elif rejected:
            reward = REJECTION_PENALTY
            metrics = {"rejected": 1.0}
        elif self.reward_mode == "step":
            reward, metrics = self._score(self.seq)

        info = {"metrics": metrics, "rejected": rejected}
        return self.observe(), float(reward), self.done, info
