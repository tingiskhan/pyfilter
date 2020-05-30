from .normalization import normalize
import torch
from typing import Union


def _matrix(weights: torch.Tensor, u: torch.Tensor):
    """
    Performs systematic resampling of a 2D array of log weights along the second axis.
    independent of the others.
    :param weights: The weights to use for resampling
    :return: Resampled indices
    """
    n = weights.shape[1]
    index_range = torch.arange(n, dtype=u.dtype, device=weights.device).unsqueeze(0)

    probs = (index_range + u) / n
    cumsum = weights.cumsum(-1)

    cumsum[..., -1] = 1.

    return torch.searchsorted(cumsum, probs)


def _vector(weights: torch.Tensor, u: torch.Tensor):
    """
    Performs systematic resampling of a 1D array log weights.
    :param weights: The weights to use for resampling
    :return: Resampled indices
    """
    n = weights.shape[0]
    probs = (torch.arange(n, dtype=u.dtype, device=weights.device) + u) / n

    cumsum = weights.cumsum(0)
    cumsum[..., -1] = 1.

    return torch.searchsorted(cumsum, probs)


def systematic(w: torch.Tensor, normalized=False, u: Union[torch.Tensor, float] = None):
    """
    Performs systematic resampling on either a 1D or 2D array.
    :param w: The weights to use for resampling
    :param normalized: Whether the data is normalized
    :param u: Parameter for overriding the sampled index, for testing
    :return: Resampled indices
    """

    shape = (1,) if w.dim() < 2 else (w.shape[0], 1)
    u = u if u is not None else (torch.empty(shape, device=w.device)).uniform_()
    w = normalize(w) if not normalized else w

    if w.dim() > 1:
        return _matrix(w, u)

    return _vector(w, u)


def multinomial(w: torch.Tensor, normalized=False):
    """
    Performs multinomial sampling.
    :param w: The weights to use for resampling
    :param normalized: Whether the data is normalized
    :return: Resampled indices
    """

    return torch.multinomial(normalize(w) if not normalized else w, w.shape[-1], replacement=True)


def residual(w: torch.Tensor, normalized=False):
    """
    Performs residual resampling. Inspired by solution provided by the package "particles" on GitHub
    authored by the user "nchopin".
    :param w: The weights to use for resampling
    :param normalized: Whether the data is normalized
    :return: Resampled indices
    """

    if w.dim() > 1:
        raise NotImplementedError('Not implemented for multidimensional arrays!')

    w = normalize(w) if not normalized else w

    # ===== Calculate the number of deterministic to get ===== #
    mw = (w.shape[-1] * w)
    floored = mw.floor()
    res = mw - floored

    # ===== Make flat ===== #
    out = torch.ones_like(w, dtype=torch.long)

    # ===== Get the indexes of those to sample ===== #
    numelems = floored.sum(-1)
    res /= numelems

    intpart = floored.long()
    ranged = torch.arange(w.shape[-1], dtype=intpart.dtype, device=w.device) * out

    # ===== Repeat the integers and transform to correct ===== #
    modded = ranged.repeat_interleave(intpart)
    aslong = numelems.long()

    out[:aslong] = modded

    if numelems == w.shape[-1]:
        return out

    out[aslong:] = torch.multinomial(res, w.shape[-1] - aslong, replacement=True)

    return out