from abc import ABC
from torch.distributions import Distribution
import torch
from functools import lru_cache
from .parameter import Parameter, size_getter
from copy import deepcopy
from ..module import Module
from typing import Tuple, Union, Callable, Dict
from ..utils import flatten


class StochasticProcessBase(Module):
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

    def sample_params(self, shape: Union[int, Tuple[int, ...], torch.Size] = None):
        """
        Samples the parameters of the model in place.
        :param shape: The shape to use
        :return: Self
        """

        for param in self.theta_dists:
            param.sample_(shape)

        self.viewify_params(shape)

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

    def copy(self):
        """
        Returns a deep copy of the object.
        :return: Copy of current instance
        """

        return deepcopy(self)


def _view_helper(p, shape):
    return p.view(*shape, *p._prior.event_shape) if len(shape) > 0 else p.view(p.shape)


class StochasticProcess(StochasticProcessBase, ABC):
    def __init__(self, theta: Tuple[object, ...], initial_dist: Union[Distribution, None],
                 increment_dist: Distribution,
                 initial_transform: Union[Callable[[Distribution, Tuple[Parameter, ...]], Distribution], None] = None):
        """
        The base class for time series.
        :param theta: The parameters governing the dynamics
        :param initial_dist: The initial distribution
        :param increment_dist: The distribution of the increments
        """

        super().__init__()

        # ===== Check distributions ===== #
        cases = (
            all(isinstance(n, Distribution) for n in (initial_dist, increment_dist)),
            (isinstance(increment_dist, Distribution) and initial_dist is None)
        )

        if not any(cases):
            raise ValueError('All must be of instance `torch.distributions.Distribution`!')

        self.initial_dist = initial_dist
        self.increment_dist = increment_dist
        self.init_transform = initial_transform

        self._mapper = {
            f'increment/{increment_dist.__class__.__name__}': self.increment_dist
        }

        if self.initial_dist is not None:
            self._mapper[f'initial/{initial_dist.__class__.__name__}'] = self.initial_dist

        # ===== Some helpers ===== #
        self._theta = None
        self._theta_vals = None

        self._inputdim = self.ndim

        # ===== Distributional parameters ===== #
        self._dist_theta = dict()
        self._org_dist = dict()

        for t, n in [('initial', self.initial_dist), ('increment', self.increment_dist)]:
            if n is None:
                continue

            parameters = dict()
            statics = dict()
            for k, v in vars(n).items():
                if k.startswith('_'):
                    continue

                if isinstance(v, Parameter) and n is self.increment_dist:
                    parameters[k] = v
                elif isinstance(v, torch.Tensor):
                    statics[k] = v

            if not not parameters:
                self._dist_theta[f'{t}/{n.__class__.__name__}'] = parameters
                self._org_dist[f'{t}/{n.__class__.__name__}'] = statics

        # ===== Regular parameters ====== #
        self.theta = tuple(Parameter(th) if not isinstance(th, Parameter) else th for th in theta)

    @property
    def distributional_theta(self) -> Dict[str, Dict[str, Parameter]]:
        """
        Returns the parameters of the distribution to re-initialize the distribution with. Mainly a helper for when
        the user passes distributions parameterized by priors.
        """

        return self._dist_theta

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, x: Tuple[Parameter, ...]):
        if self._theta is not None and len(x) != len(self._theta):
            raise ValueError('The number of parameters must be same!')
        if not all(isinstance(p, Parameter) for p in x):
            raise ValueError(f'Not all items are of instance {Parameter.__class__.__name__}')

        self._theta = x
        self.viewify_params(torch.Size([]))

    @property
    def theta_dists(self):
        return tuple(p for p in self.theta if p.trainable) + flatten((v.values() for v in self._dist_theta.values()))

    @property
    def theta_vals(self) -> Tuple[Parameter, ...]:
        """
        Returns the values of the parameters.
        """
        return self._theta_vals

    @property
    @lru_cache()
    def ndim(self):
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc - just like torch.
        """

        return len((self.initial_dist or self.increment_dist).event_shape)

    @property
    @lru_cache()
    def num_vars(self):
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process, it returns 1, and
        the number of elements in the vector/matrix.
        """

        dist = self.initial_dist or self.increment_dist
        if len(dist.event_shape) < 1:
            return 1

        prod = 1
        for i in dist.event_shape:
            prod *= i

        return prod

    def viewify_params(self, shape, in_place=True) -> Tuple[Parameter, ...]:
        shape = size_getter(shape)

        # ===== Regular parameters ===== #
        theta_vals = tuple()
        for param in self.theta:
            if param.trainable:
                var = _view_helper(param, shape)
            else:
                var = param

            theta_vals += (var,)

        if not in_place:
            return theta_vals

        # ===== Distributional parameters ===== #
        for d, dists in self.distributional_theta.items():
            temp = dict()
            temp.update(self._org_dist[d])

            for k, v in dists.items():
                temp[k] = _view_helper(v, shape)

            self._mapper[d].__init__(**temp)

        self._theta_vals = theta_vals

        return theta_vals

    def i_sample(self, shape: Union[int, Tuple[int, ...], None] = None, as_dist=False) -> torch.Tensor:
        """
        Samples from the initial distribution.
        :param shape: The number of samples
        :param as_dist: Whether to return the new value as a distribution
        :return: Samples from the initial distribution
        """

        dist = self.initial_dist.expand(size_getter(shape))

        if self.init_transform is not None:
            dist = self.init_transform(dist, *self.theta_vals)

        if as_dist:
            return dist

        return dist.sample()

    def sample_path(self, steps, samples=None, x_s=None) -> torch.Tensor:
        x_s = self.i_sample(samples) if x_s is None else x_s
        out = torch.zeros(steps, *x_s.shape, device=x_s.device, dtype=x_s.dtype)
        out[0] = x_s

        for i in range(1, steps):
            out[i] = self.propagate(out[i-1])

        return out

    def propagate_u(self, x: torch.Tensor, u: torch.Tensor):
        """
        Propagate the process conditional on both state and draws from incremental distribution.
        :param x: The previous state
        :param u: The current draws from the incremental distribution
        """

        return self._propagate_u(x, u)

    def _propagate_u(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def prop_apf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method used by APF. Propagates the state one step forward.
        :param x: The previous state
        :return: The new state
        """

        raise NotImplementedError()

    def populate_state_dict(self):
        return {
            "_theta": self.theta
        }

    def load_state_dict(self, state: Dict[str, object]):
        super(StochasticProcess, self).load_state_dict(state)
        self.viewify_params(torch.Size([]))

        return self