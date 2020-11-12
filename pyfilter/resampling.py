from .normalization import normalize
import torch
from typing import Union


def systematic(w: torch.Tensor, normalized=False, u: Union[torch.Tensor, float] = None):
    """
    Performs systematic resampling on either a 1D or 2D array.
    :param w: The weights to use for resampling
    :param normalized: Whether the data is normalized
    :param u: Parameter for overriding the sampled index, for testing
    :return: Resampled indices
    """
    is_1d = w.dim() == 1

    if is_1d:
        w = w.unsqueeze(0)

    shape = (w.shape[0], 1)
    u = u if u is not None else (torch.empty(shape, device=w.device)).uniform_()
    w = normalize(w) if not normalized else w

    n = w.shape[1]
    index_range = torch.arange(n, dtype=u.dtype, device=w.device).unsqueeze(0)

    probs = (index_range + u) / n
    cumsum = w.cumsum(-1)

    cumsum[..., -1] = 1.0
    res = torch.searchsorted(cumsum, probs)

    return res.squeeze(0) if is_1d else res


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
        raise NotImplementedError("Not implemented for multidimensional arrays!")

    w = normalize(w) if not normalized else w

    # ===== Calculate the number of deterministic to get ===== #
    mw = w.shape[-1] * w
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
