from ....resampling import systematic
import torch
from typing import Callable, Union
from ....filters import BaseFilter
from ..state import FilteringAlgorithmState


class BaseKernel(object):
    def __init__(self, resampling=systematic):
        self._resampler = resampling

    def set_resampler(self, resampler: Callable[[torch.Tensor, bool, Union[float, torch.Tensor]], torch.Tensor]):
        self._resampler = resampler

        return self

    def _update(self, filter_: BaseFilter, state: FilteringAlgorithmState, *args):
        raise NotImplementedError()

    def update(self, filter_: BaseFilter, state: FilteringAlgorithmState, *args):
        self._update(filter_, state, *args)

        return self
