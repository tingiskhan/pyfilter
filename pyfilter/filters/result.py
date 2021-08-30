import torch
from typing import List
from .state import BaseFilterState
from ..utils import TensorTuple
from ..state import StateWithTensorTuples


class FilterResult(StateWithTensorTuples):
    """
    Implements a basic object for storing log likelihoods and the filtered means of a filter algorithm.
    """

    def __init__(self, init_state: BaseFilterState, record_states: bool = False):
        super().__init__()

        self.register_buffer("_loglikelihood", init_state.get_loglikelihood())

        self.tensor_tuples["filter_means"] = TensorTuple()
        self.tensor_tuples["filter_variances"] = TensorTuple()

        self.record_states = record_states

        self._states = list()
        self._latest_state = None

        self.append(init_state)

    @property
    def loglikelihood(self) -> torch.Tensor:
        return self._buffers["_loglikelihood"]

    @property
    def filter_means(self) -> torch.Tensor:
        return self.tensor_tuples["filter_means"].values()

    @property
    def filter_variance(self) -> torch.Tensor:
        return self.tensor_tuples["filter_variances"].values()

    @property
    def states(self) -> List[BaseFilterState]:
        return self._states

    @property
    def latest_state(self) -> BaseFilterState:
        return self._latest_state

    def exchange(self, res: "FilterResult", indices: torch.Tensor):
        """
        Exchanges the specified indices of `self` with `res`.
        """

        self._loglikelihood[indices] = res.loglikelihood[indices]

        for old_tt, new_tt in zip(self.tensor_tuples.values(), res.tensor_tuples.values()):
            for old_tensor, new_tensor in zip(old_tt.tensors, new_tt.tensors):
                old_tensor[indices] = new_tensor[indices]

        self._latest_state.exchange(res.latest_state, indices)
        for ns, os in zip(res.states[:-1], self.states[:-1]):
            os.exchange(ns, indices)

        return self

    def resample(self, indices: torch.Tensor, entire_history=True):
        """
        Resamples the specified indices of `self` with `res`.
        """

        self._loglikelihood[:] = self.loglikelihood[indices]

        if entire_history:
            for tt in self.tensor_tuples.values():
                for tensor in tt.tensors:
                    tensor[:] = tensor[indices]

        self._latest_state.resample(indices)
        for s in self.states[:-1]:
            s.resample(indices)

        return self

    def append(self, state: BaseFilterState):
        self.tensor_tuples["filter_means"].append(state.get_mean())
        self.tensor_tuples["filter_variances"].append(state.get_variance())

        self._loglikelihood = self._loglikelihood + state.get_loglikelihood()
        self._latest_state = state

        if self.record_states:
            self._states.append(state)

        return self
