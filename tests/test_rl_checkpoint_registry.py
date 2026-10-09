"""Keep the updated Fig. 5 ensemble aligned with the native checkpoint registry."""

import ast
from pathlib import Path

import yaml
import pytest

from invarna.design.reward import primary_reward


ROOT = Path(__file__).resolve().parents[1]
TE_MODELS = ["w0", "w1", "w2", "final_tdc"]


def registry_paths() -> dict[str, str]:
    """Inspect the declared paths without importing the GPU inference stack in CI."""
    def relative_path(node: ast.expr) -> str:
        if isinstance(node, ast.Name) and node.id == "CHECKPOINT_ROOT":
            return ""
        assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        return str(Path(relative_path(node.left)) / ast.literal_eval(node.right))

    source = ROOT / "src/invarna/evaluation/human_te.py"
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "STAGE2_CKPTS" for target in node.targets):
            continue
        assert isinstance(node.value, ast.Dict)
        paths = {}
        for key, value in zip(node.value.keys, node.value.values):
            paths[ast.literal_eval(key)] = relative_path(value)
        return paths
    raise AssertionError("STAGE2_CKPTS registry not found")


def test_four_te_models_and_half_life_paths() -> None:
    paths = registry_paths()
    for name in ("w0", "w1", "w2"):
        assert paths[name] == f"te_student/{name}.ckpt"
    assert paths["final_tdc"] == "te_student/final_tdc.ckpt"
    assert paths["hl_new"] == "half_life/final_half_life.ckpt"
    for historical_name in ("wt", "world_new", "world_new07522"):
        assert historical_name not in paths


def test_fig5_configs_use_four_te_models() -> None:
    for gene in ("hbb", "blnk"):
        path = ROOT / "figures/fig5/design/configs" / f"{gene}_rl.yaml"
        config = yaml.safe_load(path.read_text())
        assert config["te_models"] == TE_MODELS
        assert config["half_life_model"] == "hl_new"
        assert len(set(config["te_models"])) == 4
        assert set(config["reward"]) == {
            "te_weight", "half_life_weight", "te_disagreement_weight",
            "te_disagreement_threshold", "z_clip",
        }
        assert config["filters"] == {"local_window_nt": 15}


@pytest.mark.parametrize(
    "te_scores, half_life, expected",
    [
        ([0, 0, 0, 0], 0, 0),
        ([1, 1, 1, 1], 0, 0.70),
        ([0, 0, 0, 0], 1, 0.30),
        ([0, 0, 1, 1], 0, 0.35),  # No penalty at std == 0.5.
        ([-1, -1, 1, 1], 0, -0.105),  # Disagreement remains inside the TE term.
    ],
)
def test_public_te_half_life_objective(te_scores, half_life, expected) -> None:
    assert primary_reward(te_scores, half_life) == pytest.approx(expected)
