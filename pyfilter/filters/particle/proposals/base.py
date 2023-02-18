from abc import ABC
from typing import Callable, Tuple

import torch
from stochproc.timeseries import StateSpaceModel, StructuralStochasticProcess, TimeseriesState
from torch.distributions import Distribution

from ..state import ParticleFilterPrediction
from .pre_weight_funcs import get_pre_weight_func


class Proposal(ABC):
    """
    Abstract base class for proposal objects.
    """

    def __init__(
        self, pre_weight_func: Callable[[StructuralStochasticProcess, TimeseriesState], TimeseriesState] = None
    ):
        """
        Internal initializer for :class:`Proposal`.

        Args:
            pre_weight_func: function used in :class:`APF` when weighing the particles to be propagated. A common
            choice is :math:`p(y_t | E_t[x_{t-1}])`.
        """

        super().__init__()
        self._model: StateSpaceModel = None
        self._pre_weight_func = pre_weight_func

    def set_model(self, model: StateSpaceModel):
        """
        Sets the model to be used in the proposal.

        Args:
            model (StateSpaceModel): the model to consider.
        """

        self._model = model
        self._pre_weight_func = get_pre_weight_func(self._pre_weight_func, model.hidden)

        return self

    def _weight_with_kernel(
        self, y: torch.Tensor, x_dist: Distribution, x_new: TimeseriesState, kernel: Distribution
    ) -> torch.Tensor:
        y_dist = self._model.build_density(x_new)

        return y_dist.log_prob(y) + x_dist.log_prob(x_new.value) - kernel.log_prob(x_new.value)

    def sample_and_weight(
        self, y: torch.Tensor, prediction: ParticleFilterPrediction
    ) -> Tuple[TimeseriesState, torch.Tensor]:
        """
        Method to be derived by inherited classes. Given the current observation ``y`` and prediction ``x`` of the particle filter this
        method samples new state values and weighs them accordingly.

        Args:
            y (torch.Tensor): current observation.
            x (ParticleFilterPrediction): predicted filter state.

        Returns:
            The new state together with the associated weights.
        """

        raise NotImplementedError()

    def pre_weight(self, y: torch.Tensor, x: TimeseriesState) -> torch.Tensor:
        """
        Pre-weights previous state ``x`` w.r.t. the current observation ``y``. Used in the ``APF`` when evaluating which
        candidate particles to select for propagation.

        Args:
            y (torch.Tensor): the current observation.
            x (TimeseriesState): the previous state.

        Returns:
            Returns the log weights associated with the previous state particles.
        """

        new_state = self._pre_weight_func(self._model.hidden, x)
        y_dist = self._model.build_density(new_state)

        return y_dist.log_prob(y)

    def copy(self) -> "Proposal":
        """
        Copies the proposal by returning a new instance of type ``self``.
        """

        raise NotImplementedError()
