# Sequence-only supervised fine-tuning

`train.py` calls the checkpoint-compatible recovered student trainer. Each manuscript
condition has an explicit YAML under `configs/`; the human test split is evaluation
only and checkpoint selection uses validation R².

Canonical conditions are `human_only`, `human_mouse`, `matched_random`,
`absolute_synthetic`, `wt_anchor`, and `tdc_final`.

```bash
python sft/train.py --student-config sft/configs/tdc_final.yaml
```
