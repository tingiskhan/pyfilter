import torch
from typing import Union, Iterable
from torch.nn import Module
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


KEY_PREFIX = "item"


class TensorTuple(Module):
    def __init__(self, *args):
        super().__init__()
        self._i = -1

        for a in args:
            self.append(a)

        self._register_load_state_dict_pre_hook(self._hook)

    @staticmethod
    def _make_key(i: int):
        return f"{KEY_PREFIX}_{i}"

    def append(self, x: torch.Tensor):
        self._i += 1
        self.register_buffer(self._make_key(self._i), x)

    def __getitem__(self, item: int):
        if item < 0:
            return self._buffers[self._make_key(self._i + 1 + item)]

        return self._buffers[self._make_key(item)]

    def __iter__(self):
        return (v for v in self._buffers.values())

    def __len__(self):
        return len(self._buffers)

    def values(self):
        if self._i == 0:
            return torch.empty(0)

        return torch.stack(tuple(self._buffers.values()), 0)

    def _hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for k, v in state_dict.items():
            if prefix:
                k = k.replace(prefix, "")

            if not k.startswith(KEY_PREFIX):
                continue

            self.register_buffer(k, v)

        return


def get_ess(w: torch.Tensor, normalized=False) -> torch.Tensor:
    """
    Calculates the ESS from an array of log weights.

    :param w: The log weights
    :param normalized: Whether input is normalized
    :return: The effective sample size
    """

    if not normalized:
        w = normalize(w)

    return w.sum(-1) ** 2 / (w ** 2).sum(-1)


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


def normalize(w: torch.Tensor) -> torch.Tensor:
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
    normalized[torch.isnan(ax_sum) | (ax_sum == 0.0)] = 1 / normalized.shape[-1]

    return normalized.squeeze(0) if is_1d else normalized
