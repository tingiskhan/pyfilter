from ...state import AlgorithmState
import torch
from ....filters import FilterResult
from ....utils import TensorTuple


class PMMHState(AlgorithmState):
    def __init__(self, initial_sample: torch.Tensor, filter_result: FilterResult):
        super().__init__()
        self.samples = TensorTuple(initial_sample)
        self.filter_result = filter_result

    def update(self, sample: torch.Tensor):
        self.samples.append(sample)

    def as_tensor(self):
        return torch.stack(self.samples.values(), 0)
