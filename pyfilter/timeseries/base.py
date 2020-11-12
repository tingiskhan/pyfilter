from torch.distributions import Distribution
import torch
from .parameter import Parameter
from copy import deepcopy
from ..module import Module
from typing import Tuple, Union, Callable, TypeVar
from ..utils import ShapeLike


T = TypeVar("T")


class Base(Module):
    @property
    def parameters(self) -> Tuple[Parameter, ...]:
        raise NotImplementedError()

    @property
    def trainable_parameters(self) -> Tuple[Parameter, ...]:
        raise NotImplementedError()

    def viewify_params(self, shape: Union[Tuple[int, ...], torch.Size]):
        """
        Makes views of the parameters.
        :param shape: The shape to use. Please note that this shape will be prepended to the "event shape"
        :return: Self
        """

        raise NotImplementedError()

    def parameters_to_array(self, transformed=False, as_tuple=False) -> torch.Tensor:
        raise NotImplementedError()

    def parameters_from_array(self, array: torch.Tensor, transformed=False):
        raise NotImplementedError()

    def sample_params(self, shape: ShapeLike):
        """
        Samples the parameters of the model in place.
        """

        for param in self.trainable_parameters:
            param.sample_(shape)

        self.viewify_params(shape)

        return self

    def log_prob(self, y: torch.Tensor, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Weights the process of the current state `x_t` with the previous `x_{t-1}`.
        :param y: The value at x_t
        :param x: The value at x_{t-1}
        :param u: Covariate value at time t
        """

        dist = self.define_density(x, u=u)

        return dist.log_prob(y)

    def define_density(self, x: torch.Tensor, u: torch.Tensor = None) -> Distribution:
        raise NotImplementedError()

    def propagate(self, x: torch.Tensor, as_dist=False, u: torch.Tensor = None) -> Union[Distribution, torch.Tensor]:
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous state
        :param as_dist: Whether to return the new value as a distribution
        :param u: Any covariates
        :return: Samples from the model
        """

        dist = self.define_density(x, u=u)

        if as_dist:
            return dist

        return dist.sample()

    def sample_path(
        self, steps: int, samples: Union[int, Tuple[int, ...]] = None, x_s: torch.Tensor = None, u: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Samples a trajectory from the model.
        :param steps: The number of steps
        :param samples: Number of sample paths
        :param x_s: The start value for the latent process
        :param u: Any covariates
        :return: An array of sampled values
        """

        raise NotImplementedError()

    def p_apply(self, func: Callable[[Parameter], Parameter], transformed=False):
        """
        Applies `func` to each parameter of the model inplace.
        :param func: Function to apply, must be of the structure func(param)
        :param transformed: Whether or not results from applied function are transformed variables
        :return: Self
        """

        for p in self.trainable_parameters:
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
            res = self.p_map(lambda u: u.bijected_prior.log_prob(u.t_values))
        else:
            res = self.p_map(lambda u: u.prior.log_prob(u.values))

        return sum(res)

    def p_map(self, func: Callable[[Parameter], T]) -> Tuple[T, ...]:
        """
        Applies the func to the parameters and returns a tuple of objects. Note that it is only applied to parameters
        that are distributions.
        :param func: The function to apply to parameters.
        :return: Returns tuple of values
        """

        out = tuple()
        for p in self.trainable_parameters:
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
