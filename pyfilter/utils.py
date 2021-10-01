import torch
from typing import Union, Iterable, Iterator
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from .constants import INFTY
from .typing import ShapeLike


def size_getter(shape: ShapeLike) -> torch.Size:
    """
    Utility function for aiding in generating a ``torch.Size`` object.

    Args:
        shape: The shape to use, can be ``None``, ``int`` or tuples.
    """

    if shape is None:
        return torch.Size([])
    elif isinstance(shape, torch.Size):
        return shape
    elif isinstance(shape, int):
        return torch.Size([shape])

    return torch.Size(shape)


class TensorTuple(IterableDataset):
    """
    Implements a tuple like tensor storage.
    """

    def __init__(self, *tensors):
        self.tensors = tensors

    def __iter__(self) -> Iterator[T_co]:
        for t in self.tensors:
            yield t

    def __getitem__(self, index) -> T_co:
        return self.tensors[index]

    def __add__(self, other: IterableDataset):
        return TensorTuple(*self.tensors, *other.tensors)

    def values(self) -> torch.Tensor:
        return torch.stack(self.tensors, dim=0)

    def append(self, tensor: torch.Tensor):
        self.tensors += (tensor,)

    def apply(self, fn):
        self.tensors = tuple(fn(t) for t in self.tensors)


def get_ess(weights: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """
    Calculates the ESS from an array of (log) weights.

    Args:
        weights: The (log) weights to calculate ESS for.
        normalized: Optional parameter indicating whether the weights are normalized.
    """

    if not normalized:
        weights = normalize(weights)

    return weights.sum(-1) ** 2 / (weights ** 2).sum(-1)


def choose(array: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Utility function for aiding in choosing along the first dimension of a tensor.

    Args:
        array: The tensor to choose from.
        indices: The indices to select.
    """

    if indices.dim() < 2:
        return array[indices]

    return array[torch.arange(array.shape[0], device=array.device)[:, None], indices]


def concater(*x: torch.Tensor) -> torch.Tensor:
    """
    Given an iterable of tensors, broadcast them to the same shape and stack along the last dimension.

    Args:
        x: The iterable of tensors to stack.
    """

    if isinstance(x, torch.Tensor):
        return x

    return torch.stack(torch.broadcast_tensors(*x), dim=-1)


def construct_diag_from_flat(x: torch.Tensor, base_dim: int) -> torch.Tensor:
    """
    Constructs a diagonal matrix based on ``x``. Solution found `here`_:

    Args:
        x: The diagonal of the matrix.
        base_dim: The dimension of ``x``.

    Example:
        If ``x`` is of shape ``(100, 20)`` and the dimension is 0, then we get a tensor of shape ``(100, 20, 1)``.

    .. _here: https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    """

    if base_dim == 0:
        return x.unsqueeze(-1).unsqueeze(-1)

    if base_dim == 1 and x.shape[-1] < 2:
        return x.unsqueeze(-1)

    return x.unsqueeze(-1) * torch.eye(x.shape[-1], device=x.device)


def normalize(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a 1D or 2D array of log weights.

    Args:
        weights: The log weights to normalize.
    """

    is_1d = weights.dim() == 1

    if is_1d:
        weights = weights.unsqueeze(0)

    mask = torch.isfinite(weights)
    weights[~mask] = -INFTY

    reweighed = torch.exp(weights - weights.max(-1)[0][..., None])
    normalized = reweighed / reweighed.sum(-1)[..., None]

    ax_sum = normalized.sum(1)
    normalized[torch.isnan(ax_sum) | (ax_sum == 0.0)] = 1 / normalized.shape[-1]

    return normalized.squeeze(0) if is_1d else normalized
