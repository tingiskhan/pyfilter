from copy import deepcopy
from typing import Generic, List, TypeVar, Union

import torch

from ..container import make_dequeue
from ..state import BaseResult
from .state import Correction

TCorrection = TypeVar("TCorrection", bound=Correction)
BoolOrInt = Union[bool, int]


class FilterResult(BaseResult, Generic[TCorrection]):
    """
    Implements an object for storing results when running filters.
    """

    # TODO: Add dump and load hook for all states instead of just last?
    def __init__(self, init_state: TCorrection, record_states: BoolOrInt, record_moments: BoolOrInt):
        """
        Initializes the :class:`FilterResult` class.

        Args:
            init_state: the initial state.
            record_states: parameter for whether to record all, or some of the states of the filter. Do note that
                recording all states will be very memory intensive for particle filters. See
                :func:`pyfilter.container.make_dequeue` for more details.
            record_moments: same as ``record_states`` but for the filter means and variances.
        """

        super().__init__()

        self._loglikelihood = init_state.get_loglikelihood()

        self.tensor_tuples.make_deque("filter_means", maxlen=record_moments)
        self.tensor_tuples.make_deque("filter_variances", maxlen=record_moments)

        self._states = make_dequeue(maxlen=record_states)
        self.append(init_state)

    @property
    def loglikelihood(self) -> torch.Tensor:
        r"""
        Returns the current estimate of the total log likelihood, :math:`\log p(y_{1:t})`.
        """

        return self._loglikelihood

    @property
    def filter_means(self) -> torch.Tensor:
        """
        Returns the estimated filter means, of shape
        ``(timesteps, [batch shape], latent dimension)``.
        """

        return self.tensor_tuples.get_as_tensor("filter_means")

    @property
    def filter_variance(self) -> torch.Tensor:
        """
        Returns the estimated filter variances, of shape
        ``(timesteps, [batch shape], latent dimension)``.
        """

        return self.tensor_tuples.get_as_tensor("filter_variances")

    @property
    def states(self) -> List[TCorrection]:
        return list(self._states)

    @property
    def latest_state(self) -> TCorrection:
        return self._states[-1]

    def exchange(self, other: "FilterResult[TCorrection]", mask: torch.Tensor):
        """
        Exchanges the states and tensor tuples with ``res`` at ``mask``. Note that this is only relevant for filters
        that have been run in parallel.

        Args:
            other: the object to exchange states and tensor tuples with.
            mask: mask specifying which values to exchange.
        """

        self._loglikelihood[mask] = other.loglikelihood[mask]

        for old_tt, new_tt in zip(self.tensor_tuples.values(), other.tensor_tuples.values()):
            for old, new in zip(old_tt, new_tt):
                old[mask] = new[mask]

        for ns, os in zip(other.states, self.states):
            os.exchange(ns, mask)

        return self

    def resample(self, indices: torch.IntTensor, entire_history=True):
        """
        Resamples tensor tuples and states.

        Args:
            indices: the indices to select.
            entire_history: optional parameter for whether to resample entire history or not. If ``False``, we ignore
                resampling the tensor tuples.
        """

        self._loglikelihood.copy_(self._loglikelihood[indices])

        if entire_history:
            for tt in self.tensor_tuples.values():
                for tens in tt:
                    tens.copy_(tens[indices])

        for s in self.states:
            s.resample(indices)

        return self

    def append(self, state: TCorrection):
        """
        Appends state to ``self``.

        Args:
            state: The state to append.
        """

        self.tensor_tuples["filter_means"].append(state.get_mean())
        self.tensor_tuples["filter_variances"].append(state.get_variance())

        self._loglikelihood.add_(state.get_loglikelihood())
        self._states.append(state)

        return self

    def state_dict(self):
        """
        Converts ``self`` to a dictionary. Currently onnly handles the latest state.
        """

        res = super(FilterResult, self).state_dict()

        res["state"] = self.latest_state.state_dict()
        res["log_likelihood"] = self.loglikelihood

        return res

    def load_state_dict(self, state_dict):
        """
        Loads state from existing state dictionary.

        Args:
            state_dict: state dictionary to load from.
        """

        super().load_state_dict(state_dict)
        self._loglikelihood = state_dict["log_likelihood"]

        assert len(self.states) == 1, "Can only handle case when we have 1 state!"
        self.latest_state.load_state_dict(state_dict["state"])

        return

    def __repr__(self):
        return f"FilterResult(ll: {self._loglikelihood.__repr__()}, num_observations: {self.filter_means.shape[0]})"

    def copy(self) -> "FilterResult":
        """
        Performs a deep copy of self.
        """

        return deepcopy(self)
