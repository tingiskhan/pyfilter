import torch


def _vector(w: torch.Tensor):
    mask = torch.isfinite(w)
    w[~mask] = float('-inf')

    reweighed = torch.exp(w - w.max())

    normalized = reweighed / reweighed.sum()
    if normalized.sum() == 0:
        normalized[:] = 1 / normalized.shape[-1]

    return normalized


def _matrix(w: torch.Tensor):
    mask = torch.isfinite(w)
    w[~mask] = float('-inf')

    reweighed = torch.exp(w - w.max(-1)[0][..., None])
    normalized = reweighed / reweighed.sum(-1)[..., None]
    normalized[torch.isnan(normalized.sum(1))] = 1 / normalized.shape[-1]

    return normalized


def normalize(w: torch.Tensor):
    """
    Normalizes a 1D or 2D array of log weights.
    :param w: The weights
    :return: Normalized weights
    """

    if w.dim() > 1:
        return _matrix(w)

    return _vector(w)