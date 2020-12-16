from .normalization import normalize
import torch
from torch.distributions import Distribution
import numpy as np
from typing import Union, Tuple, Iterable
from torch.nn import Module

TensorOrDist = Union[Distribution, torch.Tensor]
ArrayType = Union[float, int, np.ndarray]
ShapeLike = Union[int, Tuple[int, ...], torch.Size]


def size_getter(shape: ShapeLike) -> torch.Size:
    if shape is None:
        return torch.Size([])
    elif isinstance(shape, torch.Size):
        return shape
    elif isinstance(shape, int):
        return torch.Size([shape])

    return torch.Size(shape)


class TensorList(Module):
    def __init__(self, *args):
        super().__init__()
        self._i = -1

        for a in args:
            self.append(a)

    def append(self, x: torch.Tensor):
        self._i += 1
        self.register_buffer(str(self._i), x)

    def __getitem__(self, item: int):
        if item < 0:
            return self._buffers[str(self._i + 1 + item)]

        return self._buffers[str(item)]

    def __iter__(self):
        return (v for v in self._buffers.values())

    def __len__(self):
        return len(self._buffers)

    def values(self):
        if self._i == 0:
            return torch.empty(0)

        return torch.stack(tuple(self._buffers.values()), 0)


class ModuleList(Module):
    def __init__(self, *args):
        super().__init__()
        self._i = -1

        for a in args:
            self.append(a)

    def append(self, x: Module):
        self._i += 1
        self.add_module(str(self._i), x)

    def __getitem__(self, item: int):
        if item < 0:
            return self._modules[str(self._i + 1 + item)]

        return self._modules[str(item)]

    def __iter__(self):
        return (v for v in self._modules.values())

    def __len__(self):
        return len(self._modules)

    def values(self):
        return tuple(self._modules.values())


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
