#!/usr/bin/env python3
"""CPU smoke test of copied Fig. 5 environment/PPO with a TE+HL proxy evaluator.

The deterministic evaluator is intentionally smoke-only: it checks the integration
chain without loading manuscript predictors or claiming manuscript-score reproduction.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from Bio.Seq import Seq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "figures/fig5/design/source"
DEFAULT_CONFIG = ROOT / "figures/fig5/design/configs/fig5_rl_smoke.yaml"
sys.path.insert(0, str(ROOT / "src"))

from invarna.design.reward import primary_reward, robust_te_reward


def load_copied_implementation():
    sys.path.insert(0, str(SOURCE))
    try:
        from agent_ppo import PPOAgent
        from env_sequence import SequenceEnv
    finally:
        sys.path.pop(0)
    return PPOAgent, SequenceEnv


class SmokeTEHLEvaluator:
    """Deterministic TE/HL proxy used only to exercise the RL integration path."""

    def __init__(self, wild_type: str, te_weight: float, half_life_weight: float):
        self.wild_type = wild_type
        self.te_weight = te_weight
        self.half_life_weight = half_life_weight
        # Four TE-like outputs and one HL-like output, with fixed toy coefficients.
        self.coefficients = np.random.default_rng(1234).normal(size=(5, len(wild_type), 4))
        self.wt_scores = self._scores(wild_type)

    def _scores(self, sequence: str) -> np.ndarray:
        indices = np.array(["ACGT".index(base) for base in sequence])
        return self.coefficients[:, np.arange(len(sequence)), indices].sum(axis=1)

    def __call__(self, sequence: str, *_unused) -> tuple[float, dict[str, float]]:
        z_scores = np.clip((self._scores(sequence) - self.wt_scores) / 2.0, -3.0, 3.0)
        te_delta = robust_te_reward(z_scores[:4])
        half_life_delta = float(z_scores[4])
        reward = primary_reward(
            z_scores[:4], half_life_delta, self.te_weight, self.half_life_weight
        )
        return float(reward), {
            "smoke_te_delta": float(te_delta),
            "smoke_half_life_delta": float(half_life_delta),
            "reward_contrib_te": float(self.te_weight * te_delta),
            "reward_contrib_stability": float(self.half_life_weight * half_life_delta),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    records: dict[str, list[str]] = {}
    current = None
    for line in (ROOT / cfg["reference_sequence"]).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            records[current] = []
        elif current is not None:
            records[current].append(line)
    def get_single_record(suffix: str) -> str:
        matches = ["".join(parts) for name, parts in records.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"WT FASTA must contain exactly one *{suffix} record")
        return matches[0]

    wt = get_single_record("_cds")
    utr5 = get_single_record("_utr5")
    utr3 = get_single_record("_utr3")
    ppo = cfg["ppo"]
    objective = cfg["objectives"]
    PPOAgent, SequenceEnv = load_copied_implementation()
    evaluator = SmokeTEHLEvaluator(
        wt,
        float(objective["te_weight"]),
        float(objective["half_life_weight"]),
    )
    assert evaluator(wt)[0] == 0.0
    env = SequenceEnv(
        init_seq=wt,
        genename="NGF",
        editregion="cds",
        max_steps=int(ppo["max_steps_per_episode"]),
        reward_fn=evaluator,
        reward_mode="terminal",
        device="cpu",
    )
    assert env.action_dim > 0
    assert all(index not in {0, env.n_codons - 1} for index, _ in env.action_map)

    agent = PPOAgent(
        seq_len=env.L,
        action_dim=env.action_dim,
        device="cpu",
        gamma=float(ppo["gamma"]),
        lam=float(ppo["gae_lambda"]),
        clip_eps=float(ppo["clip_epsilon"]),
        lr=float(ppo["learning_rate"]),
        vf_coef=float(ppo["value_coefficient"]),
        ent_coef=float(ppo["entropy_coefficient"]),
        max_grad_norm=float(ppo["max_gradient_norm"]),
        epochs=int(ppo["epochs"]),
        minibatch_size=int(ppo["minibatch_size"]),
        d_model=int(ppo["model_width"]),
    )
    before = {name: value.detach().clone() for name, value in agent.net.state_dict().items()}
    obs_buf, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
    candidates: list[tuple[str, float, dict[str, float]]] = []
    obs = env.reset()
    for _ in range(int(ppo["rollout_steps"])):
        action, logprob, value = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        obs_buf.append(obs.cpu().numpy())
        actions.append(action)
        logprobs.append(logprob)
        rewards.append(reward)
        dones.append(float(done))
        values.append(value)
        obs = next_obs
        if done:
            sequence = "".join("ACGT"[int(base)] for base in obs)
            candidates.append((sequence, reward, info["metrics"]))
            obs = env.reset()

    with torch.no_grad():
        _, last_value_tensor = agent.net(obs.unsqueeze(0))
    advantages, returns = agent._compute_gae(
        rewards, values, dones, float(last_value_tensor.item())
    )
    loss = agent.update(
        {
            "obs": np.stack(obs_buf),
            "actions": np.asarray(actions),
            "logprobs": np.asarray(logprobs, dtype=np.float32),
            "returns": returns,
            "advantages": advantages,
        }
    )
    changed = any(not torch.equal(before[name], value) for name, value in agent.net.state_dict().items())
    assert changed and np.isfinite(loss)

    candidate, reward, metrics = max(candidates, key=lambda item: item[1])
    sequence_reward, score_metrics = evaluator(candidate)
    assert np.isclose(
        sequence_reward,
        score_metrics["reward_contrib_te"] + score_metrics["reward_contrib_stability"],
    )
    assert len(candidate) == len(wt)
    assert candidate[:3] == wt[:3] and candidate[-3:] == wt[-3:]
    assert str(Seq(candidate).translate()) == str(Seq(wt).translate())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "smoke_actor_critic.pt"
    torch.save(agent.net.state_dict(), checkpoint)
    restored = PPOAgent(
        seq_len=env.L,
        action_dim=env.action_dim,
        device="cpu",
        epochs=1,
        minibatch_size=int(ppo["minibatch_size"]),
        d_model=int(ppo["model_width"]),
    )
    restored.net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for name, value in agent.net.state_dict().items():
        assert torch.equal(value, restored.net.state_dict()[name])

    csv_path = args.output_dir / "ngf_smoke_candidate.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant", "reward", "smoke_te_delta", "smoke_half_life_delta",
                "cds_sequence", "mrna_sequence",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "NGF_SMOKE",
                "reward": reward,
                "smoke_te_delta": metrics["smoke_te_delta"],
                "smoke_half_life_delta": metrics["smoke_half_life_delta"],
                "cds_sequence": candidate,
                "mrna_sequence": utr5 + candidate + utr3,
            }
        )
    fasta_path = args.output_dir / "ngf_smoke_candidate.fasta"
    fasta_path.write_text(f">NGF_SMOKE synonymous_CDS reward={reward:.6f}\n{candidate}\n")
    print(
        "PASS: SequenceEnv -> TE/HL smoke evaluator -> rollout -> GAE -> PPO update "
        "-> checkpoint save/load -> synonymous candidate -> CSV/FASTA export"
    )
    print(f"action_dim={env.action_dim} loss={loss:.6f} reward={reward:.6f}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
