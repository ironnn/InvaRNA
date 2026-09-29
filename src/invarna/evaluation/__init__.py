"""Inference and publication benchmark helpers."""

__all__ = ["InvaRNAPredictor"]


def __getattr__(name):
    if name == "InvaRNAPredictor":
        from .inference_api import InvaRNAPredictor

        return InvaRNAPredictor
    raise AttributeError(name)
