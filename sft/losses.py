"""Regression loss used by the recovered student trainer."""

from torch import Tensor
from torch.nn.functional import mse_loss


def regression_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """Return the exact mean-squared-error objective used during SFT."""
    return mse_loss(prediction.view(-1), target.view(-1))
