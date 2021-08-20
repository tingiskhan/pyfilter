from ....resampling import systematic
import torch
from typing import Callable, Union
from ....filters import BaseFilter
from ..state import SequentialAlgorithmState


class BaseKernel(object):
    """
    Base object for mutating parameters in a sequential particle algorithms.
    """

    def __init__(self, resampling=systematic):
        self._resampler = resampling

    def set_resampler(self, resampler: Callable[[torch.Tensor, bool, Union[float, torch.Tensor]], torch.Tensor]):
        self._resampler = resampler

        return self

    def _update(self, filter_: BaseFilter, state: SequentialAlgorithmState, *args):
        raise NotImplementedError()

    def update(self, filter_: BaseFilter, state: SequentialAlgorithmState, *args):
        self._update(filter_, state, *args)

        return self
