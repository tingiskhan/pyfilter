import torch
from torch.nn import Module
from typing import List
from .state import BaseState
from ..utils import TensorTuple


class FilterResult(Module):
    def __init__(self, init_state: BaseState, record_states: bool = False):
        """
        Implements a basic object for storing log likelihoods and the filtered means of a filter algorithm.
        """
        super().__init__()

        self.register_buffer("_loglikelihood", init_state.get_loglikelihood())

        self._filter_means = TensorTuple()
        self._filter_vars = TensorTuple()

        self.record_states = record_states

        self._states = list()
        self._latest_state = None

        self.append(init_state)

    @property
    def loglikelihood(self) -> torch.Tensor:
        return self._buffers["_loglikelihood"]

    @property
    def filter_means(self) -> torch.Tensor:
        return self._filter_means.values()

    @property
    def filter_variance(self) -> torch.Tensor:
        return self._filter_vars.values()

    @property
    def states(self) -> List[BaseState]:
        return self._states

    @property
    def latest_state(self) -> BaseState:
        return self._latest_state

    def exchange(self, res: "FilterResult", indices: torch.Tensor):
        """
        Exchanges the specified indices of self with res.
        """

        self._loglikelihood[indices] = res.loglikelihood[indices]

        # TODO: Not the best...
        for old_fm, new_fm in zip(self._filter_means, res._filter_means):
            old_fm[indices] = new_fm[indices]

        for old_var, new_var in zip(self._filter_vars, res._filter_vars):
            old_var[indices] = new_var[indices]

        self._latest_state.exchange(res.latest_state, indices)
        for ns, os in zip(res.states[:-1], self.states[:-1]):
            os.exchange(ns, indices)

        return self

    def resample(self, indices: torch.Tensor, entire_history=True):
        """
        Resamples the specified indices of self with res.
        """

        self._loglikelihood[:] = self.loglikelihood[indices]

        if entire_history:
            for mean, var in zip(self._filter_means, self._filter_vars):
                mean[:] = mean[indices]
                var[:] = var[indices]

        self._latest_state.resample(indices)
        for s in self.states[:-1]:
            s.resample(indices)

        return self

    def append(self, state: BaseState):
        self._filter_means.append(state.get_mean())
        self._filter_vars.append(state.get_variance())

        self._loglikelihood = self._loglikelihood + state.get_loglikelihood()
        self._latest_state = state

        if self.record_states:
            self._states.append(state)

        return self
