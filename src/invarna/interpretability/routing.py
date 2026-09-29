"""Aggregate per-position Motif-MoE routing weights."""

import torch


def summarize_routing(weights: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """Return mean expert usage for routing weights shaped ``(..., experts)``."""
    if attention_mask is not None:
        selected = weights[attention_mask.bool()]
    else:
        selected = weights.reshape(-1, weights.shape[-1])
    return selected.float().mean(dim=0)


def capture_forward_routing(model, inference_call):
    """Run ``inference_call`` and return prediction plus gate probabilities.

    The released model exposes the forward routing network as a module named
    ``gate_fwd``. Discovery by module name avoids depending on a legacy package
    path stored in older checkpoints.
    """
    candidates = [(name, module) for name, module in model.named_modules() if name.endswith("gate_fwd")]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one gate_fwd module, found {[name for name, _ in candidates]}")
    captured = []
    handle = candidates[0][1].register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    try:
        prediction = inference_call()
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("gate_fwd hook produced no routing logits")
    return prediction, torch.softmax(torch.cat(captured, dim=0).float(), dim=-1)


@torch.no_grad()
def infer_aligned_routing(predictor, aligned_sequences, batch_size: int = 64):
    """Return ``gate_fwd`` softmax probabilities for already framed sequences."""
    outputs = []
    device = predictor.device
    for offset in range(0, len(aligned_sequences), batch_size):
        sequences = aligned_sequences[offset:offset + batch_size]
        encoded = predictor.tokenizer(sequences, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to(device)
        input_ids = torch.where(input_ids == predictor.unk_id, predictor.pad_id, input_ids)
        attention_mask = (~torch.isin(input_ids, predictor.ids_to_mask)).long()

        def inference_call():
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                return predictor.model(input_ids, attention_mask=attention_mask)

        _, probabilities = capture_forward_routing(predictor.model, inference_call)
        outputs.append(probabilities.cpu())
    return torch.cat(outputs, dim=0).numpy()
