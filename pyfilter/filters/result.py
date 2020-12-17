import torch
from torch.nn import Module
from typing import Tuple
from .state import BaseState
from ..utils import TensorList, ModuleList


class FilterResult(Module):
    def __init__(self, init_state: BaseState):
        """
        Implements a basic object for storing log likelihoods and the filtered means of a filter.
        """
        super().__init__()

        self.register_buffer("_loglikelihood", init_state.get_loglikelihood())

        self._filter_means = TensorList()
        self._states = ModuleList(init_state)

    @property
    def loglikelihood(self) -> torch.Tensor:
        return self._loglikelihood

    @property
    def filter_means(self) -> torch.Tensor:
        return self._filter_means.values()

    @property
    def states(self) -> Tuple[BaseState, ...]:
        return self._states.values()

    @property
    def latest_state(self) -> BaseState:
        return self._states[-1]

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

        for ns, os in zip(res.states, self.states):
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

        for s in self._states:
            s.resample(inds)

        return self

    def append(self, state: BaseState, only_latest=True):
        self._filter_means.append(state.get_mean())

        self._loglikelihood += state.get_loglikelihood()
        if only_latest:
            self._states = ModuleList(state)
        else:
            self._states.append(state)

        return self