import torch
from ...state import ParticleState
from ....filters import FilterResult
from ....utils import TensorTuple


class PMMHState(ParticleState):
    def __init__(self, initial_sample: torch.Tensor, filter_result: FilterResult):
        super().__init__(filter_result)
        self.samples = TensorTuple(initial_sample)

    def update(self, sample: torch.Tensor):
        self.samples.append(sample)

    def as_tensor(self):
        return torch.stack(self.samples.values(), 0)
