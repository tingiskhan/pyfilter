from ...state import AlgorithmState
import torch
from ....filters import FilterResult


class PMMHState(AlgorithmState):
    def __init__(self, initial_sample: torch.Tensor, filter_result: FilterResult):
        self.samples = [initial_sample]
        self.filter_result = filter_result

    def update(self, sample: torch.Tensor):
        self.samples.append(sample)

    def as_tensor(self):
        return torch.stack(self.samples)
