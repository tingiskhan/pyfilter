import torch
from torch.nn import Module
from typing import List
from .state import BaseState
from ..utils import TensorTuple


class FilterResult(Module):
    def __init__(self, init_state: BaseState):
        """
        Implements a basic object for storing log likelihoods and the filtered means of a filter algorithm.
        """
        super().__init__()

        self.register_buffer("_loglikelihood", init_state.get_loglikelihood())

        self._filter_means = TensorTuple()
        self._latest_state = init_state
        self._states = list()  # TODO: Can't really seem to be able to serialize these

    @property
    def loglikelihood(self) -> torch.Tensor:
        return self._buffers["_loglikelihood"]

    @property
    def filter_means(self) -> torch.Tensor:
        return self._filter_means.values()

    @property
    def states(self) -> List[BaseState]:
        return self._states

    @property
    def latest_state(self) -> BaseState:
        return self._latest_state

    def exchange(self, res, inds: torch.Tensor):
        """
        Exchanges the specified indices of self with res.
        :param res: The other filter result
        :type res: FilterResult
        :param inds: The indices
        """

        # ===== Loglikelihood ===== #
        self._loglikelihood[inds] = res.loglikelihood[inds]

        # ===== Filter means ====== #
        # TODO: Not the best...
        for old_fm, new_fm in zip(self._filter_means, res._filter_means):
            old_fm[inds] = new_fm[inds]

        self._latest_state.exchange(res.latest_state, inds)
        for ns, os in zip(res.states[:-1], self.states[:-1]):
            os.exchange(ns, inds)

        return self

    def resample(self, inds: torch.Tensor, entire_history=True):
        """
        Resamples the specified indices of self with res.
        """

        self._loglikelihood = self.loglikelihood[inds]

        if entire_history:
            for mean in self._filter_means:
                mean[:] = mean[inds]

        self._latest_state.resample(inds)
        for s in self.states[:-1]:
            s.resample(inds)

        return self

    def append(self, state: BaseState, only_latest=True):
        self._filter_means.append(state.get_mean())

        self._loglikelihood += state.get_loglikelihood()
        self._latest_state = state

        if not only_latest:
            self._states.append(state)

        return self
