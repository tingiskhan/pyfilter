import torch
from ...state import FilterAlgorithmState
from ....filters import FilterResult
from ....utils import TensorTuple


class PMMHState(FilterAlgorithmState):
    """
    State for PMMH algorithm.
    """

    def __init__(self, initial_sample: torch.Tensor, filter_result: FilterResult):
        super().__init__(filter_result)
        self.samples = TensorTuple(initial_sample)

        self._register_state_dict_hook(TensorTuple.dump_hook)
        self._register_load_state_dict_pre_hook(lambda *args: TensorTuple.load_hook(self, *args))

    def update(self, sample: torch.Tensor):
        self.samples.append(sample)
