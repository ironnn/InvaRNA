#!/usr/bin/env python
# encoding: utf-8
"""
convert_weights_local.py — Convert a LucaOne FTP `pytorch.pth` checkpoint into
local HuggingFace format (no huggingface.co needed) for full-parameter fine-tuning.

Based on the official `huggingface` branch script
(https://github.com/LucaOne/LucaOne -> src/scripts/convert_weights.py),
adapted to convert a single checkpoint with explicit, machine-local paths.

Usage:
  conda run -n lucaone python convert_weights_local.py \
      --ckpt  /path/to/lucaone-gene/pytorch.pth \
      --out   assets/checkpoints/external/pretrained/models/lucaone_hf_ckpts/LucaOne-gene-step36.8M \
      --src   external_models/LucaOne/source/src/lucaone
"""
import os
import json
import shutil
import argparse
import torch
from lucaone import LucaGPLMConfig, LucaGPLMForMaskedLM, LucaGPLMTokenizer


def convert(ckpt_path, save_path, src_dir):
    os.makedirs(save_path, exist_ok=True)

    print(f"Loading original state_dict from {ckpt_path} ...")
    old_state_dict = torch.load(ckpt_path, map_location="cpu")

    shared_weight = old_state_dict.get("embed_tokens.weight")
    if shared_weight is not None:
        actual_vocab_size = shared_weight.shape[0]
        actual_hidden_size = shared_weight.shape[1]
        print(f"Inferred vocab_size={actual_vocab_size}, hidden_size={actual_hidden_size}")
    else:
        actual_vocab_size, actual_hidden_size = 39, 2560
        print("Warning: embed_tokens.weight not found, using defaults (39, 2560).")

    config = LucaGPLMConfig(
        vocab_size=actual_vocab_size,
        hidden_size=actual_hidden_size,
        num_hidden_layers=20,
        num_attention_heads=40,
        tie_word_embeddings=False,
    )

    # merge values from the checkpoint's own config.json (next to pytorch.pth)
    ckpt_cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    if os.path.exists(ckpt_cfg_path):
        ckpt_config = json.load(open(ckpt_cfg_path, "r", encoding="utf-8"))
        for k, v in ckpt_config.items():
            if hasattr(config, k) and k not in ["id2label", "label2id", "max_position_embeddings"]:
                if getattr(config, k) != v:
                    print(f"  config update: {k} -> {v}")
                    setattr(config, k, v)

    config.auto_map = {
        "AutoConfig": "configuration_lucaone.LucaGPLMConfig",
        "AutoTokenizer": ["tokenization_lucaone.LucaGPLMTokenizer", None],
        "AutoModel": "modeling_lucaone.LucaGPLMModel",
        "AutoModelForMaskedLM": "modeling_lucaone.LucaGPLMForMaskedLM",
        "AutoModelForSequenceClassification": "modeling_lucaone.LucaGPLMForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_lucaone.LucaGPLMForTokenClassification",
    }

    model = LucaGPLMForMaskedLM(config)
    tokenizer = LucaGPLMTokenizer(vocab_type="gene_prot")

    print("Mapping weights ...")
    new_state_dict = {}
    for k, v in old_state_dict.items():
        if k.startswith("lm_head"):
            if k in ("lm_head.weight", "lm_head.decoder.weight"):
                new_state_dict["lm_head.decoder.weight"] = v
            else:
                new_state_dict[k] = v
        elif any(k.startswith(p) for p in [
            "contact_head", "hidden_layer_list", "hidden_act_list",
            "classifier_dropout_list", "classifier_list", "output_list", "loss_fct_list",
        ]):
            continue
        elif k.startswith("layers."):
            new_state_dict[f"lucaone.encoder.{k}"] = v
        elif k.startswith("last_layer_norm."):
            new_state_dict[f"lucaone.encoder.{k}"] = v
        elif k.startswith("embed_"):
            new_state_dict[f"lucaone.embeddings.{k}"] = v

    if shared_weight is not None:
        new_state_dict["lucaone.embeddings.embed_tokens.weight"] = shared_weight.clone()
        new_state_dict["lm_head.decoder.weight"] = shared_weight.clone()

    print("Loading state_dict into model (strict=True) ...")
    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    print("State dict loaded successfully!")

    print(f"Saving model to {save_path} ...")
    model.save_pretrained(save_path)

    print(f"Saving tokenizer to {save_path} ...")
    tokenizer.tokenizer_class = "LucaGPLMTokenizer"
    tokenizer.save_pretrained(save_path)

    tok_cfg = os.path.join(save_path, "tokenizer_config.json")
    if os.path.exists(tok_cfg):
        t = json.load(open(tok_cfg, "r", encoding="utf-8"))
        t["tokenizer_class"] = "LucaGPLMTokenizer"
        t["auto_map"] = {"AutoTokenizer": ["tokenization_lucaone.LucaGPLMTokenizer", None]}
        json.dump(t, open(tok_cfg, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("Copying remote-code files for trust_remote_code loading ...")
    for f in ["__init__.py", "configuration_lucaone.py", "modeling_lucaone.py", "tokenization_lucaone.py"]:
        src_file = os.path.join(src_dir, f)
        if os.path.exists(src_file):
            shutil.copy(src_file, save_path)
            print(f"  copied {f}")
        else:
            print(f"  WARNING: {src_file} not found")

    print("-" * 60)
    print("CONVERSION COMPLETE!")
    print(f"Model Path: {save_path}")
    print(f"Load with: AutoModel.from_pretrained('{save_path}', trust_remote_code=True)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to FTP pytorch.pth")
    ap.add_argument("--out", required=True, help="output HF checkpoint dir")
    ap.add_argument("--src", required=True, help="dir holding the 4 lucaone *.py remote-code files")
    args = ap.parse_args()
    convert(args.ckpt, args.out, args.src)
