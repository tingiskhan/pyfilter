from typing import Callable, TypeVar, Tuple, Generic

from torch.distributions import TransformedDistribution

from .filters.particle.state import ParticleFilterState
from .inference.sequential.state import SequentialAlgorithmState
from .state import BaseState
import torch.nn

from stochproc.timeseries import StateSpaceModel

T = TypeVar("T", bound=BaseState)


__all__ = ["Collector", "MeanCollector", "Standardizer", "ParameterPosterior"]


class Collector(Generic[T]):
    """
    Defines a collector object that is registered as a hook on a ``torch.nn.Module`` and calculates some quantity
    that is saved to the state object's ``.tensor_tuples`` attribute.
    """

    def __init__(self, name: str, f: Callable[[torch.nn.Module, torch.Tensor, T], None]):
        """
        Initializes the ``Collector`` class.

        Args:
            name: The name to assign to the ``tensor_tuple``.
            f: The function to use when calculating statics.
        """

        self._name = name
        self._f = f

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, module: torch.nn.Module, inp: Tuple[torch.Tensor, T], out: T):
        if self._name not in out.tensor_tuples:
            out.tensor_tuples[self._name] = tuple()

        out.tensor_tuples[self._name] += (self._f(module, inp[0], out),)


class MeanCollector(Collector[SequentialAlgorithmState]):
    """
    Defines a collector that calculates averaged mean, used in sequential algorithms.
    """

    @staticmethod
    def _mean(algorithm, y, new_state: SequentialAlgorithmState):
        latest_means = new_state.filter_state.latest_state.get_mean()

        return new_state.normalized_weights() @ latest_means

    def __init__(self):
        super().__init__(name="filter_means", f=self._mean)


class Standardizer(Collector[SequentialAlgorithmState]):
    """
    Defines a collector that calculates the residuals.
    """

    @staticmethod
    def _standardize(ssm: StateSpaceModel, y, x):
        y_std = y
        dist = ssm.observable.build_density(x)

        if not isinstance(dist, TransformedDistribution):
            raise NotImplementedError(f"Can't standardize for '{dist.__class__.__name__}'")

        for t in reversed(dist.transforms):
            y_std = t.inv(y_std)

        return y_std

    def _fun(self, algorithm, y, new_state: SequentialAlgorithmState):
        filter_state = new_state.filter_state.latest_state

        residuals = self._standardize(algorithm.filter.ssm, y, filter_state.get_timeseries_state())
        if isinstance(filter_state, ParticleFilterState):
            residuals = (filter_state.normalized_weights() * residuals).sum(dim=-1)

        return new_state.normalized_weights() @ residuals

    def __init__(self):
        super().__init__(name="standardized", f=self._fun)


class ParameterPosterior(Collector[SequentialAlgorithmState]):
    """
    Collects the first moment of the parameter posterior.
    """

    def _mean_var(self, algorithm, y, new_state: SequentialAlgorithmState):
        parameters = algorithm.filter.ssm.concat_parameters(constrained=self._constrained)

        return new_state.normalized_weights() @ parameters

    def __init__(self, constrained=True):
        """
        Initializes the ``ParameterPosterior`` collector.

        Args:
            constrained: Whether to record constrained or non-constrained.
        """

        super(ParameterPosterior, self).__init__(name="parameter_means", f=self._mean_var)
        self._constrained = constrained
