import torch
from typing import Union, Iterable, Iterator
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from .constants import INFTY
from .typing import ShapeLike


def size_getter(shape: ShapeLike) -> torch.Size:
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

    @staticmethod
    def dump_hook(obj, state_dict, prefix, local_metadata):
        for k, v in vars(obj).items():
            if not isinstance(v, TensorTuple):
                continue

            state_dict[k] = getattr(obj, k)

    @staticmethod
    def load_hook(obj, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for k, v in vars(obj).items():
            if not isinstance(v, TensorTuple):
                continue

            setattr(obj, k, state_dict.pop(k))

def get_ess(weights: torch.Tensor, normalized=False) -> torch.Tensor:
    """
    Calculates the ESS from an array of log weights.
    """

    if not normalized:
        weights = normalize(weights)

    return weights.sum(-1) ** 2 / (weights ** 2).sum(-1)


def choose(array: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.dim() < 2:
        return array[indices]

    return array[torch.arange(array.shape[0], device=array.device)[:, None], indices]


def loglikelihood(w: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    maxw, _ = w.max(-1)

    if weights is None:
        temp = torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw)).mean(-1).log()
    else:
        temp = (weights * torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw))).sum(-1).log()

    return maxw + temp


def concater(*x: Union[Iterable[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x

    return torch.stack(torch.broadcast_tensors(*x), dim=-1)


def construct_diag_from_flat(x: torch.Tensor, base_dim: int) -> torch.Tensor:
    """
    Constructs a diagonal matrix based on batched data. Solution found here:
    https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    """

    if base_dim == 0:
        return x.unsqueeze(-1).unsqueeze(-1)

    if base_dim == 1 and x.shape[-1] < 2:
        return x.unsqueeze(-1)

    return x.unsqueeze(-1) * torch.eye(x.shape[-1], device=x.device)


def normalize(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a 1D or 2D array of log weights.
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
