from torch.nn import Module
from ..filters import FilterResult


class AlgorithmState(Module):
    pass


class ParticleState(AlgorithmState):
    def __init__(self, filter_state: FilterResult):
        super().__init__()
        self.filter_state = filter_state

    def copy(self, filter_state: FilterResult):
        return ParticleState(filter_state)
