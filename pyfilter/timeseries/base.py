from torch.distributions import Distribution
import torch
from functools import lru_cache
from .parameter import Parameter
from ..utils.utils import add_dimensions


def _get_shape(x, ndim):
    """
    Gets the shape to generate samples for.
    :param x: The tensor
    :type x: torch.Tensor|float
    :param ndim: The dimensions
    :type ndim: int
    :rtype: tuple[int]
    """

    if not isinstance(x, torch.Tensor):
        return ()

    return x.shape if ndim < 2 else x.shape[1:]


class BaseModel(object):
    def __init__(self, initial, funcs, theta, noise):
        """
        This object is to serve as a base class for the timeseries models.
        :param initial: The functions governing the initial dynamics of the process
        :type initial: tuple[callable]
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple[callable]
        :param theta: The parameters governing the dynamics
        :type theta: tuple[Distribution]|tuple[torch.Tensor]|tuple[float]
        :param noise: The noise governing the noise process
        :type noise: tuple[Distribution]
        """

        self.f0, self.g0 = initial
        self.f, self.g = funcs
        self._theta = tuple(Parameter(th) for th in theta)

        cases = (
            all(isinstance(n, Distribution) for n in noise),
            (isinstance(noise[-1], Distribution) and noise[0] is None)
        )

        if not any(cases):
            raise ValueError('All must be of instance `torch.distributions.Distribution`!')

        self.noise0, self.noise = noise

    @property
    def theta(self):
        """
        Returns the parameters of the model.
        :rtype: tuple[Parameter]
        """

        return self._theta

    @property
    def theta_dists(self):
        """
        Returns the indices for parameter are distributions.
        :rtype: tuple[Parameter]
        """

        return tuple(p for p in self.theta if p.trainable)

    @property
    def theta_vals(self):
        """
        Returns the values of the parameters
        :rtype: tuple[float|torch.Tensor]
        """

        return tuple(th.values for th in self.theta)

    @property
    @lru_cache()
    def ndim(self):
        """
        Returns the dimension of the process.
        :return: Dimension of process
        :rtype: int
        """
        shape = self.noise.mean.shape
        if len(shape) < 1:
            return 1

        return shape[0]

    def i_mean(self, params=None):
        """
        Calculates the mean of the initial distribution.
        :param params: Used for overriding the parameters
        :type params: tuple[torch.Tensor]|tuple[float]
        :return: The mean of the initial distribution
        :rtype: torch.Tensor
        """

        return self.f0(*(params or self.theta_vals))

    def i_scale(self, params=None):
        """
        Calculates the scale of the initial distribution.
        :param params: Used for overriding the parameters
        :type params: tuple[torch.Tensor]|tuple[float]
        :return: The scale of the initial distribution
        :rtype: torch.Tensor
        """

        return self.g0(*(params or self.theta_vals))

    def i_weight(self, x, params=None):
        """
        Weights the process of the initial state.
        :param x: The state at `x_0`.
        :type x: torch.Tensor
        :param params: Used for overriding the parameters
        :type params: tuple[torch.Tensor]|tuple[float]
        :return: The log-weights
        :rtype: torch.Tensor
        """

        loc, scale = self.i_mean(params), self.i_scale(params)

        if self.ndim < 2:
            rescaled = (x - loc) / scale
        else:
            # TODO: Might not work
            rescaled = scale.inverse().dot(x - loc)

        return self.noise0.log_prob(rescaled)

    def mean(self, x, params=None):
        """
        Calculates the mean of the process conditional on the previous state and current parameters.
        :param x: The state of the process.
        :type x: torch.Tensor|float
        :param params: Used for overriding the parameters
        :type params: tuple[torch.Tensor]|tuple[float]
        :return: The mean
        :rtype: torch.Tensor|float
        """

        return self.f(x, *(params or self.theta_vals))

    def scale(self, x, params=None):
        """
        Calculates the scale of the process conditional on the current state and parameters.
        :param x: The state of the process
        :type x: torch.Tensor|float
        :param params: Used for overriding the parameters
        :type params: tuple of np.ndarray|float|int
        :return: The scale
        :rtype: torch.Tensor|float
        """

        return self.g(x, *(params or self.theta_vals))

    def weight(self, y, x, params=None):
        """
        Weights the process of the current state `x_t` with the previous `x_{t-1}`. Used whenever the proposal
        distribution is different from the underlying.
        :param y: The value at x_t
        :type y: torch.Tensor|float
        :param x: The value at x_{t-1}
        :type x: torch.Tensor|float
        :param params: Used of overriding the parameters
        :type params: tuple[torch.Tensor]|tuple[float]
        :return: The log-weights
        :rtype: torch.Tensor|float
        """
        loc, scale = self.mean(x, params=params), self.scale(x, params=params)

        rescaled = (add_dimensions(y, loc.dim()) - loc) / scale

        if self.ndim > 1:
            rescaled = rescaled.permute(*range(1, rescaled.dim() - 1), 0)
            log_scale = scale.abs().log().sum(0)
        else:
            log_scale = scale.log()

        return self.noise.log_prob(rescaled) - log_scale

    def i_sample(self, shape=None, params=None):
        """
        Samples from the initial distribution.
        :param shape: The number of samples
        :type shape: int|tuple[int]
        :param params: Any parameters to use instead
        :type params: tuple[torch.Tensor]|tuple[float]
        :return: Samples from the initial distribution
        :rtype: torch.Tensor|float
        """

        loc, scale = self.i_mean(params), self.i_scale(params)
        rndshape = ((shape,) if isinstance(shape, int) else shape) or torch.Size()
        eps = self.noise0.sample(rndshape)

        if self.ndim > 1:
            eps = eps.permute(-1, *range(eps.dim() - 1))
            loc = add_dimensions(loc, eps.dim())
            scale = add_dimensions(scale, eps.dim())

        return loc + scale * eps

    def propagate(self, x, params=None):
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous state
        :type x: torch.Tensor|float
        :param params: Used for overriding the parameters
        :type params: tuple[torch.Tensor]|tuple[float]
        :return: Samples from the model
        :rtype: torch.Tensor|float
        """

        loc, scale = self.mean(x, params), self.scale(x, params)

        if isinstance(self, Observable):
            shape = _get_shape(loc, self.ndim)
        else:
            shape = _get_shape(x, self.ndim)

        eps = self.noise.sample(shape)
        if self.ndim > 1:
            eps = eps.permute(-1, *range(eps.dim() - 1))
            loc = add_dimensions(loc, eps.dim())
            scale = add_dimensions(scale, eps.dim())

        return loc + scale * eps

    def sample(self, steps, samples=None, params=None):
        """
        Samples a trajectory from the model.
        :param steps: The number of steps
        :type steps: int
        :param samples: Number of sample paths
        :type samples: int
        :param params: Used for overriding the parameters
        :type params: tuple[torch.Tensor]|tuple[float]
        :return: An array of sampled values
        :rtype: torch.Tensor
        """

        shape = steps, self.ndim

        if samples is not None:
            shape += (*((samples,) if not isinstance(samples, (list, tuple)) else samples),)

        out = torch.zeros(shape)
        out[0] = self.i_sample(shape=samples, params=params)

        for i in range(1, steps):
            out[i] = self.propagate(out[i-1], params=params)

        return out

    def p_apply(self, func, transformed=False):
        """
        Applies `func` to each parameter of the model "inplace", i.e. manipulates `self.theta`.
        :param func: Function to apply, must be of the structure func(param)
        :type func: callable
        :param transformed: Whether or not results from applied function are transformed variables
        :type transformed: bool
        :return: Instance of self
        :rtype: BaseModel
        """

        for p in self.theta_dists:
            if transformed:
                p.t_values = func(p)
            else:
                p.values = func(p)

        return self

    def p_prior(self):
        """
        Calculates the prior log-likelihood of the current values.
        :return: The prior of the current parameter values
        :rtype: np.ndarray|float
        """

        return sum(self.p_map(lambda u: u.logpdf(u.values)))

    def p_map(self, func, default=None):
        """
        Applies the func to the parameters and returns a tuple of objects. Note that it is only applied to parameters
        that are distributions.
        :param func: The function to apply to parameters.
        :type func: callable
        :param default: What to set those parameters that aren't distributions to. If `None`, sets to the current value
        :type default: np.ndarray|float|int
        :return: Returns tuple of values
        :rtype: tuple[Parameter]
        """

        out = tuple()
        for p in self.theta:
            if isinstance(p, Distribution):
                out += (func(p),)
            else:
                out += (default if default is not None else p,)

        return out


class Observable(BaseModel):
    def __init__(self, funcs, theta, noise):
        """
        Object for defining the observable part of an HMM.
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple of callable
        :param theta: The parameters governing the dynamics
        :type theta: tuple of np.ndarray|tuple of float|tuple of Distribution
        :param noise: The noise governing the noise process
        :type noise: Distribution
        """
        super().__init__((None, None), funcs, theta, (None, noise))