from .normalization import normalize
import torch
from typing import Union, Tuple, Iterable
from torch.nn import Module


ShapeLike = Union[int, Tuple[int, ...], torch.Size]


class TensorList(Module):
    def __init__(self):
        super().__init__()
        self._i = 0

    def append(self, x: torch.Tensor):
        self.register_buffer(str(self._i), x)
        self._i += 1

    def __getitem__(self, item: int):
        return self._buffers[str(item)]


def get_ess(w: torch.Tensor, normalized=False):
    """
    Calculates the ESS from an array of log weights.
    :param w: The log weights
    :param normalized: Whether input is normalized
    :return: The effective sample size
    """

    if not normalized:
        w = normalize(w)

    return w.sum(-1) ** 2 / (w ** 2).sum(-1)


def choose(array: torch.Tensor, indices: torch.Tensor):
    if indices.dim() < 2:
        return array[indices]

    return array[torch.arange(array.shape[0], device=array.device)[:, None], indices]


def loglikelihood(w: torch.Tensor, weights: torch.Tensor = None):
    maxw, _ = w.max(-1)

    # ===== Calculate the second term ===== #
    if weights is None:
        temp = torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw)).mean(-1).log()
    else:
        temp = (weights * torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw))).sum(-1).log()

    return maxw + temp


def concater(*x: Union[Iterable[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x

    return torch.stack(torch.broadcast_tensors(*x), dim=-1)


def construct_diag(x: torch.Tensor):
    """
    Constructs a diagonal matrix based on batched data. Solution found here:
    https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    Do note that it only considers the last axis.
    """

    if x.dim() < 1:
        return x
    elif x.shape[-1] < 2:
        return x.unsqueeze(-1)
    elif x.dim() < 2:
        return torch.diag(x)

    return x.unsqueeze(-1) * torch.eye(x.shape[-1], device=x.device)


class TempOverride(object):
    def __init__(self, obj: object, attr: str, new_vals: object):
        """
        Implements a temporary override of attribute of an object.
        :param obj: An object
        :param attr: The attribute to override
        :param new_vals: The new values
        """
        self._obj = obj
        self._attr = attr
        self._new_vals = new_vals
        self._old_vals = None

    def __enter__(self):
        self._old_vals = getattr(self._obj, self._attr)
        setattr(self._obj, self._attr, self._new_vals)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self._obj, self._attr, self._old_vals)

        return False
