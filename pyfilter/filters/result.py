import torch
from torch.nn import Module
from typing import List
from .state import BaseState
from ..utils import AppendableTensorList


class FilterResult(Module):
    def __init__(self, init_state: BaseState):
        """
        Implements a basic object for storing log likelihoods and the filtered means of a filter.
        """
        super().__init__()

        self.register_buffer("_loglikelihood", init_state.get_loglikelihood())

        self._filter_means = AppendableTensorList()
        self._latest_state = init_state
        self._states = list()   # TODO: Can't really seem to be able to serialize these

    @property
    def loglikelihood(self) -> torch.Tensor:
        return self._loglikelihood

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
        for old_fm, new_fm in zip(self.filter_means, res.filter_means):
            old_fm[inds] = new_fm[inds]

        for ns, os in zip(res.states, self.states):
            os.exchange(ns, inds)

        return self

    def resample(self, inds: torch.Tensor, entire_history=True):
        """
        Resamples the specified indices of self with res.
        """

        self._loglikelihood = self.loglikelihood[inds]

        if entire_history:
            for mean in self.filter_means:
                mean[:] = mean[inds]

        for s in self._states:
            s.resample(inds)

        return self

    def append(self, state: BaseState, only_latest=True):
        self._filter_means.append(state.get_mean())

        self._loglikelihood += state.get_loglikelihood()
        self._latest_state = state

        if not only_latest:
            self._states.append(state)

        return self
