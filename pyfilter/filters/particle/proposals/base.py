import torch
from torch.distributions import Distribution
from typing import Callable
from abc import ABC
from .pre_weight_funcs import get_pre_weight_func
from ....timeseries import StochasticProcess, StateSpaceModel, NewState


class Proposal(ABC):
    """
    Abstract base class for proposal objects.
    """

    def __init__(self, pre_weight_func: Callable[[StochasticProcess, NewState], NewState] = None):
        """
        Initializes the proposal object.

        Args:
            pre_weight_func: Function used in the ``APF`` when weighing the particles to be propagated. A common choice
                is :math:`p(y_t | E_t[x_{t-1}])`, where :math:`E_t[...}` denotes the expected value of the stochastic
                process at time ``t`` using the values at ``t-1``.
        """

        super().__init__()
        self._model = None  # type: StateSpaceModel
        self._pre_weight_func = pre_weight_func

    def set_model(self, model: StateSpaceModel):
        """
        Sets the model to be used in the proposal.

        Args:
            model: The model to be used.
        """

        self._model = model
        self._pre_weight_func = get_pre_weight_func(self._pre_weight_func, model.hidden)

        return self

    def _weight_with_kernel(self, y: torch.Tensor, x_new: NewState, kernel: Distribution) -> torch.Tensor:
        y_dist = self._model.observable.build_density(x_new)
        return y_dist.log_prob(y) + x_new.dist.log_prob(x_new.values) - kernel.log_prob(x_new.values)

    def sample_and_weight(self, y: torch.Tensor, x: NewState) -> (NewState, torch.Tensor):
        """
        Method to be derived by inherited classes. Given the current observation ``y`` and previous state ``x``, this
        method samples new state values and weighs them accordingly.

        Args:
            y: The current observation.
            x: The previous state.

        Returns:
            The new state together with the associated weights.
        """

        raise NotImplementedError()

    def pre_weight(self, y: torch.Tensor, x: NewState) -> torch.Tensor:
        """
        Pre-weights previous state ``x`` w.r.t. the current observation ``y``. Used in the ``APF`` when evaluating which
        candidate particles to select for propagation.

        Args:
            y: The current observation.
            x: The previous state.

        Returns:
            Returns the log weights associated with the previous state particles.
        """

        new_state = self._pre_weight_func(self._model.hidden, x)
        y_state = self._model.observable.propagate(new_state)

        return y_state.dist.log_prob(y)
