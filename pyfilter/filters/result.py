import torch
from typing import List, TypeVar, Generic, Union
from stochproc.container import make_dequeue
from .state import FilterState
from ..state import BaseResult

TState = TypeVar("TState", bound=FilterState)
BoolOrInt = Union[bool, int]


class FilterResult(BaseResult, Generic[TState]):
    """
    Implements an object for storing results when running filters.
    """

    # TODO: Add dump and load hook for all states instead of just last?
    def __init__(self, init_state: TState, record_states: BoolOrInt, record_moments: BoolOrInt):
        """
        Initializes the ``FilterResult`` object.

        Args:
            init_state: The initial state.
            record_states: Parameter for whether to record all, or some of the states of the filter. Do note that
                recording all states will be very memory intensive for particle filters. See
                ``pyfilter.container.make_dequeue`` for more details.
            record_moments: Same as ``record_states`` but for the filter means and variances.
        """

        super().__init__()

        self._loglikelihood = init_state.get_loglikelihood()

        self.tensor_tuples.make_deque("filter_means", maxlen=record_moments)
        self.tensor_tuples.make_deque("filter_variances", maxlen=record_moments)

        self._states = make_dequeue(maxlen=record_states)
        self.append(init_state)

    @property
    def loglikelihood(self) -> torch.Tensor:
        """
        Returns the current estimate of the total log likelihood, :math:`\\log{p(y_{1:t})}`.
        """

        return self._loglikelihood

    @property
    def filter_means(self) -> torch.Tensor:
        """
        Returns the estimated filter means, of shape
        ``(number of timesteps, [number of parallel filters], dimension of latent space)``.
        """

        return self.tensor_tuples.get_as_tensor("filter_means")

    @property
    def filter_variance(self) -> torch.Tensor:
        """
        Returns the estimated filter variances, of shape
        ``(number of timesteps, [number of parallel filters], dimension of latent space)``.
        """

        return self.tensor_tuples.get_as_tensor("filter_variances")

    @property
    def states(self) -> List[TState]:
        return list(self._states)

    @property
    def latest_state(self) -> TState:
        return self._states[-1]

    def exchange(self, res: "FilterResult[TState]", mask: torch.Tensor):
        """
        Exchanges the states and tensor tuples with ``res`` at ``mask``. Note that this is only relevant for filters
        that have been run in parallel.

        Args:
            res: the object to exchange states and tensor tuples with.
            mask: mask specifying which values to exchange.
        """

        self._loglikelihood[mask] = res.loglikelihood[mask]

        for old_tt, new_tt in zip(self.tensor_tuples.values(), res.tensor_tuples.values()):
            for old, new in zip(old_tt, new_tt):
                old[mask] = new[mask]

        for ns, os in zip(res.states, self.states):
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

    def append(self, state: TState):
        """
        Appends state to ``self``, wraps around the ``__call__`` method of ``torch.nn.Module``.

        Args:
            state: The state to append.
        """

        self.tensor_tuples["filter_means"].append(state.get_mean())
        self.tensor_tuples["filter_variances"].append(state.get_variance())

        # TODO: Might be able to do this better?
        self._loglikelihood = self._loglikelihood + state.get_loglikelihood()

        self._states.append(state)

        return self
