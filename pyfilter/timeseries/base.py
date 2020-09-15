from torch.distributions import Distribution
import torch
from .parameter import Parameter
from copy import deepcopy
from ..module import Module
from typing import Tuple, Union, Callable, Iterable
from ..utils import StackedObject


class Base(Module):
    @property
    def theta(self) -> Tuple[Parameter, ...]:
        """
        Returns the parameters of the model.
        """

        raise NotImplementedError()

    @property
    def theta_dists(self) -> Tuple[Parameter, ...]:
        """
        Returns the parameters that are distributions.
        """

        raise NotImplementedError()

    def viewify_params(self, shape: Union[Tuple[int, ...], torch.Size]):
        """
        Makes views of the parameters.
        :param shape: The shape to use. Please note that this shape will be prepended to the "event shape"
        :return: Self
        """

        raise NotImplementedError()

    def update_parameters(self, params: Iterable[torch.Tensor], transformed=True):
        raise NotImplementedError()

    def parameters_as_matrix(self, transformed=True) -> StackedObject:
        raise NotImplementedError()

    def sample_params(self, shape: Union[int, Tuple[int, ...], torch.Size] = None):
        """
        Samples the parameters of the model in place.
        :param shape: The shape to use
        :return: Self
        """

        for param in self.theta_dists:
            param.sample_(shape)

        return self

    def log_prob(self, y: torch.Tensor, x: torch.Tensor):
        """
        Weights the process of the current state `x_t` with the previous `x_{t-1}`. Used whenever the proposal
        distribution is different from the underlying.
        :param y: The value at x_t
        :param x: The value at x_{t-1}
        :return: The log-weights
        """

        return self._log_prob(y, x)

    def _log_prob(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def propagate(self, x: torch.Tensor, as_dist=False) -> Union[Distribution, torch.Tensor]:
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous state
        :param as_dist: Whether to return the new value as a distribution
        :return: Samples from the model
        """

        return self._propagate(x, as_dist)

    def _propagate(self, x, as_dist=False):
        raise NotImplementedError()

    def sample_path(self, steps: int, samples: Union[int, Tuple[int, ...]] = None,
                    x_s: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Samples a trajectory from the model.
        :param steps: The number of steps
        :param samples: Number of sample paths
        :param x_s: The start value for the latent process
        :return: An array of sampled values
        """

        raise NotImplementedError()

    def p_apply(self, func: Callable[[Parameter], Parameter], transformed=False):
        """
        Applies `func` to each parameter of the model "inplace", i.e. manipulates `self.theta`.
        :param func: Function to apply, must be of the structure func(param)
        :param transformed: Whether or not results from applied function are transformed variables
        :return: Self
        """

        for p in self.theta_dists:
            if transformed:
                p.t_values = func(p)
            else:
                p.values = func(p)

        return self

    def p_prior(self, transformed=True) -> torch.Tensor:
        """
        Calculates the prior log-likelihood of the current values.
        :param transformed: If you use an unconstrained proposal you need to use `transformed=True`
        :return: The prior of the current parameter values
        """

        if transformed:
            prop1 = 'transformed_dist'
            prop2 = 't_values'
        else:
            prop1 = 'dist'
            prop2 = 'values'

        return sum(self.p_map(lambda u: getattr(u, prop1).log_prob(getattr(u, prop2))))

    def p_map(self, func: Callable[[Parameter], object]) -> Tuple[object, ...]:
        """
        Applies the func to the parameters and returns a tuple of objects. Note that it is only applied to parameters
        that are distributions.
        :param func: The function to apply to parameters.
        :return: Returns tuple of values
        """

        out = tuple()
        for p in self.theta_dists:
            out += (func(p),)

        return out

    def copy(self, view_shape=torch.Size([])):
        """
        Returns a deep copy of the object.
        :return: Copy of current instance
        """
        res = deepcopy(self)
        res.viewify_params(view_shape)

        return res
