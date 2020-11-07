import torch
from .constants import INFTY


def normalize(w: torch.Tensor):
    """
    Normalizes a 1D or 2D array of log weights.
    :param w: The weights
    :return: Normalized weights
    """

    is_1d = w.dim() == 1

    if is_1d:
        w = w.unsqueeze(0)

    mask = torch.isfinite(w)
    w[~mask] = -INFTY

    reweighed = torch.exp(w - w.max(-1)[0][..., None])
    normalized = reweighed / reweighed.sum(-1)[..., None]

    ax_sum = normalized.sum(1)
    normalized[torch.isnan(ax_sum) | (ax_sum == 0.)] = 1 / normalized.shape[-1]

    return normalized if not is_1d else normalized.squeeze(0)