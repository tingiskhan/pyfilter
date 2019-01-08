from torch.distributions import Distribution
import torch
from functools import lru_cache
from .parameter import Parameter
from ..utils import isfinite, concater
from .statevariable import tensor_caster


def finite_decorator(func):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)

        mask = isfinite(out)

        if (~mask).any():
            out[~mask] = -2e20

        return out

    return wrapper


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

    return x.shape if ndim < 2 else x.shape[:-1]


def parameter_caster(ndim, *args):
    """
    Wrapper for re-casting parameters to correct sizes.
    :rtype: torch.Tensor
    """
    targs = tuple()
    for a in args:
        vals = a.values

        if a.trainable and vals.dim() > 0:
            diff = ndim - vals.dim()
            vals = vals.view(*vals.shape, *(diff * (1,)))

        targs += (vals,)

    return targs


def init_caster(func):
    def wrapper(obj):
        return concater(func(obj))

    return wrapper


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
        self._inputdim = self.ndim

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

    @init_caster
    def i_mean(self):
        """
        Calculates the mean of the initial distribution.
        :return: The mean of the initial distribution
        :rtype: torch.Tensor
        """

        return self.f0(*self.theta_vals)

    @init_caster
    def i_scale(self):
        """
        Calculates the scale of the initial distribution.
        :return: The scale of the initial distribution
        :rtype: torch.Tensor
        """

        return self.g0(*self.theta_vals)

    def i_weight(self, x):
        """
        Weights the process of the initial state.
        :param x: The state at `x_0`.
        :type x: torch.Tensor
        :return: The log-weights
        :rtype: torch.Tensor
        """

        loc, scale = self.i_mean(), self.i_scale()

        if self.ndim < 2:
            rescaled = (x - loc) / scale
        else:
            # TODO: Might not work
            rescaled = scale.inverse().dot(x - loc)

        return self.noise0.log_prob(rescaled)

    @tensor_caster
    def f_val(self, x):
        """
        Evaluates the drift part of the process.
        :param x: The state of the process.
        :type x: torch.Tensor|float
        :return: The mean
        :rtype: torch.Tensor|float
        """

        return self.f(x, *parameter_caster(x.dim() - bool(self._inputdim > 1), *self.theta))

    @tensor_caster
    def g_val(self, x):
        """
        Evaluates the diffusion part of the process.
        :param x: The state of the process.
        :type x: torch.Tensor|float
        :return: The mean
        :rtype: torch.Tensor|float
        """

        return self.g(x, *parameter_caster(x.dim() - bool(self._inputdim > 1), *self.theta))

    def mean(self, x):
        """
        Calculates the mean of the process conditional on the previous state and current parameters.
        :param x: The state of the process.
        :type x: torch.Tensor|float
        :return: The mean
        :rtype: torch.Tensor|float
        """

        return self.f_val(x)

    def scale(self, x):
        """
        Calculates the scale of the process conditional on the current state and parameters.
        :param x: The state of the process
        :type x: torch.Tensor|float
        :return: The scale
        :rtype: torch.Tensor|float
        """

        return self.g_val(x)

    @finite_decorator
    def weight(self, y, x):
        """
        Weights the process of the current state `x_t` with the previous `x_{t-1}`. Used whenever the proposal
        distribution is different from the underlying.
        :param y: The value at x_t
        :type y: torch.Tensor|float
        :param x: The value at x_{t-1}
        :type x: torch.Tensor|float
        :return: The log-weights
        :rtype: torch.Tensor|float
        """
        loc, scale = self.mean(x), self.scale(x)

        rescaled = (y - loc) / scale

        if self.ndim > 1:
            log_scale = scale.abs().log().sum(-1)
        else:
            log_scale = scale.log()

        return self.noise.log_prob(rescaled) - log_scale

    def i_sample(self, shape=None):
        """
        Samples from the initial distribution.
        :param shape: The number of samples
        :type shape: int|tuple[int]
        :return: Samples from the initial distribution
        :rtype: torch.Tensor|float
        """

        loc, scale = self.i_mean(), self.i_scale()
        rndshape = ((shape,) if isinstance(shape, int) else shape) or torch.Size()
        eps = self.noise0.sample(rndshape)

        return loc + scale * eps

    def propagate(self, x):
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous state
        :type x: torch.Tensor|float
        :return: Samples from the model
        :rtype: torch.Tensor|float
        """

        loc, scale = self.mean(x), self.scale(x)

        if isinstance(self, Observable):
            shape = _get_shape(loc, self.ndim)
        else:
            shape = _get_shape(x, self.ndim)

        eps = self.noise.sample(shape)

        return loc + scale * eps

    def sample(self, steps, samples=None):
        """
        Samples a trajectory from the model.
        :param steps: The number of steps
        :type steps: int
        :param samples: Number of sample paths
        :type samples: int
        :return: An array of sampled values
        :rtype: torch.Tensor
        """

        shape = steps, self.ndim

        if samples is not None:
            shape += (*((samples,) if not isinstance(samples, (list, tuple)) else samples),)

        out = torch.zeros(shape)
        out[0] = self.i_sample(shape=samples)

        for i in range(1, steps):
            out[i] = self.propagate(out[i-1])

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

        return sum(self.p_map(lambda u: u.dist.log_prob(u.values), default=torch.zeros(1)))

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
            if p.trainable:
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