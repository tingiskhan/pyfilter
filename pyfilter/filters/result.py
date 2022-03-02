import torch
from typing import List, TypeVar, Generic
from .state import FilterState
from ..container import make_dequeue, BoolOrInt, add_right
from ..state import BaseState

TState = TypeVar("TState", bound=FilterState)


class FilterResult(BaseState, Generic[TState]):
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

        self.register_buffer("_loglikelihood", init_state.get_loglikelihood())

        self.tensor_tuples["filter_means"] = torch.tensor([])
        self.tensor_tuples["filter_variances"] = torch.tensor([])

        self._record_moments = record_moments

        self._states = make_dequeue(maxlen=record_states)
        self.append(init_state)

        self._register_state_dict_hook(self._state_dump_hook)
        self._register_load_state_dict_pre_hook(self._state_load_hook)

    @property
    def loglikelihood(self) -> torch.Tensor:
        """
        Returns the current estimate of the total log likelihood, :math:`\\log{p(y_{1:t})}`.
        """

        return self._buffers["_loglikelihood"]

    @property
    def filter_means(self) -> torch.Tensor:
        """
        Returns the estimated filter means, of shape
        ``(number of timesteps, [number of parallel filters], dimension of latent space)``.
        """

        return self.tensor_tuples["filter_means"]

    @property
    def filter_variance(self) -> torch.Tensor:
        """
        Returns the estimated filter variances, of shape
        ``(number of timesteps, [number of parallel filters], dimension of latent space)``.
        """

        return self.tensor_tuples["filter_variances"]

    @property
    def states(self) -> List[TState]:
        return list(self._states)

    @property
    def latest_state(self) -> TState:
        return self._states[-1]

    def exchange(self, res: "FilterResult[TState]", indices: torch.Tensor):
        """
        Exchanges the states and tensor tuples with ``res`` at ``indices``. Note that this is only relevant for filters
        that have been run in parallel.

        Args:
            res: The object to exchange states and tensor tuples with.
            indices: Mask specifying which values to exchange.
        """

        self._loglikelihood[indices] = res.loglikelihood[indices]

        for old_tt, new_tt in zip(self.tensor_tuples.values(), res.tensor_tuples.values()):
            for old_tensor, new_tensor in zip(old_tt.tensors, new_tt.tensors):
                old_tensor[indices] = new_tensor[indices]

        for ns, os in zip(res.states, self.states):
            os.exchange(ns, indices)

        return self

    def resample(self, indices: torch.Tensor, entire_history=True):
        """
        Resamples tensor tuples and states.

        Args:
            indices: The indices to select.
            entire_history: Optional parameter for whether to resample entire history or not. If ``False``, we ignore
                resampling the tensor tuples.
        """

        self._loglikelihood[:] = self.loglikelihood[indices]

        if entire_history:
            for tt in self.tensor_tuples.values():
                for tensor in tt.tensors:
                    tensor[:] = tensor[indices]

        for s in self.states:
            s.resample(indices)

        return self

    def forward(self, state: TState):
        with torch.no_grad():
            add_right(self.tensor_tuples["filter_means"], state.get_mean())
            add_right(self.tensor_tuples["filter_variances"], state.get_variance())

        # TODO: Might be able to do this better?
        self._loglikelihood = self._loglikelihood + state.get_loglikelihood()

        self._states.append(state)

        return self

    def append(self, state: TState):
        """
        Appends state to ``self``, wraps around the ``__call__`` method of ``torch.nn.Module``.

        Args:
            state: The state to append.
        """

        return self.__call__(state)

    @staticmethod
    def _state_dump_hook(self: "FilterResult[TState]", state_dict, prefix, local_metadata):
        state_dict["latest_state"] = self.latest_state.state_dict(prefix=prefix)

    def _state_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # TODO: Might have use prefix?
        self.latest_state.load_state_dict(state_dict.pop(prefix + "latest_state"))
