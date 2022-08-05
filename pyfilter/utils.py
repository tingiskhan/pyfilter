import torch
from functools import wraps
from .constants import INFTY


def get_ess(weights: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """
    Calculates the ESS from an array of (log) weights.

    Args:
        weights: the log weights to calculate ESS for.
        normalized: optional parameter specifying whether the weights are normalized.
    """

    if not normalized:
        weights = normalize(weights)

    return weights.sum(-1) ** 2 / (weights ** 2).sum(-1)


def construct_diag_from_flat(x: torch.Tensor, event_shape: torch.Size) -> torch.Tensor:
    """
    Constructs a diagonal matrix based on ``x``. Solution found `here`_:

    Args:
        x: the diagonal of the matrix.
        event_shape: event shape of the process for which matrix is to be constructed.

    Example:
        If ``x`` is of shape ``(100, 20)`` and the dimension is 0, then we get a tensor of shape ``(100, 20, 1)``.

    .. _here: https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    """

    assert len(event_shape) <= 1

    eye = torch.eye(event_shape.numel(), device=x.device, dtype=x.dtype)

    if len(event_shape) == 0:
        diag = x.view(*x.shape, 1, 1)
    else:
        diag = x.unsqueeze(-1)

    return eye * diag


def normalize(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a 1D or 2D array of log weights.

    Args:
        weights: the log weights to normalize.
    """

    is_1d = weights.dim() == 1

    if is_1d:
        weights = weights.unsqueeze(0)

    mask = ~weights.isfinite()
    weights[mask] = -INFTY

    reweighed = torch.exp(weights - weights.max(-1)[0].unsqueeze(-1))
    normalized = reweighed / reweighed.sum(-1).unsqueeze(-1)

    ax_sum = normalized.sum(dim=-1)
    normalized[ax_sum == 0.0] = 1 / normalized.shape[-1]

    return normalized.squeeze(0) if is_1d else normalized


def is_documented_by(original):
    """
    Wrapper for function for copying doc strings of functions. See `this`_ reference.

    Args:
        original: the original function to copy the docs from.

    .. _this: https://softwareengineering.stackexchange.com/questions/386755/sharing-docstrings-between-similar-functions
    """

    @wraps(original)
    def wrapper(target):
        target.__doc__ = original.__doc__

        return target

    return wrapper
