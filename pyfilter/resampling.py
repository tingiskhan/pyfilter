from .utils import normalize
import torch
from typing import Union


def systematic(w: torch.Tensor, normalized=False, u: Union[torch.Tensor, float] = None) -> torch.Tensor:
    """
    Performs systematic resampling on either a 1D or 2D array.

    Args:
        w: The (log) weights to use for resampling.
        normalized: Whether the weights are normalized are not.
        u: Parameter for overriding the sampled index, only used for testing.
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


def multinomial(w: torch.Tensor, normalized=False) -> torch.Tensor:
    """
    Performs multinomial sampling.

    Args:
        w: See ``systematic``.
        normalized: See ``systematic``.
    """

    return torch.multinomial(normalize(w) if not normalized else w, w.shape[-1], replacement=True)


def residual(w: torch.Tensor, normalized=False) -> torch.Tensor:
    """
    Performs residual resampling. Inspired by solution provided by the package "particles" on GitHub
    authored by the user "nchopin".

    Args:
        w: See ``systematic``.
        normalized: See ``systematic``.
    """

    if w.dim() > 1:
        raise NotImplementedError("Not implemented for multidimensional arrays!")

    w = normalize(w) if not normalized else w

    mw = w.shape[-1] * w
    floored = mw.floor()
    res = mw - floored

    out = torch.ones_like(w, dtype=torch.long)

    numelems = floored.sum(-1)
    res /= numelems

    intpart = floored.long()
    ranged = torch.arange(w.shape[-1], dtype=intpart.dtype, device=w.device) * out

    modded = ranged.repeat_interleave(intpart)
    aslong = numelems.long()

    out[:aslong] = modded

    if numelems == w.shape[-1]:
        return out

    out[aslong:] = torch.multinomial(res, w.shape[-1] - aslong, replacement=True)

    return out
