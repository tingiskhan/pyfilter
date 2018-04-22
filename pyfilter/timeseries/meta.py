import numpy as np
from ..distributions.continuous import Distribution
from ..utils.utils import resizer


class Base(object):
    def __init__(self, initial, funcs, theta, noise, q=None):
        """
        This object is to serve as a base class for the timeseries models.
        :param initial: The functions governing the initial dynamics of the process
        :type initial: tuple of callable
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple of callable
        :param theta: The parameters governing the dynamics
        :type theta: tuple of Distribution|tuple of np.ndarray|tuple of float|tuple of int
        :param noise: The noise governing the noise process
        :type noise: (Distribution, Distribution)
        :param q: The correlation of the noise processes
        :type q: numpy.ndarray
        """

        self.f0, self.g0 = initial
        self.f, self.g = funcs
        self.theta = theta
        self._o_theta = theta           # We save the original inputs if is of `Distribution`
        self.noise0, self.noise = noise
        self.q = q

    @property
    def ndim(self):
        """
        Returns the dimension of the process.
        :return: Dimension of process
        :rtype: int
        """
        return self.noise.ndim

    @property
    def priors(self):
        """
        Returns the priors of the time series.
        :return: The priors of the model
        :rtype: tuple of Distribution
        """
        return self._o_theta

    def i_mean(self, params=None):
        """
        Calculates the mean of the initial distribution.
        :param params: Used for overriding the parameters
        :type params: tuple of np.ndarray|float|int
        :return: The mean of the initial distribution
        :rtype: np.ndarray|float|int
        """

        return resizer(self.f0(*(params or self.theta)))

    def i_scale(self, params=None):
        """
        Calculates the scale of the initial distribution.
        :param params: Used for overriding the parameters
        :type params: tuple of np.ndarray|float|int
        :return: The scale of the initial distribution
        :rtype: np.ndarray|float|int
        """

        return resizer(self.g0(*(params or self.theta)))

    def i_weight(self, x, params=None):
        """
        Weights the process of the initial state.
        :param x: The state at `x_0`.
        :type x: np.ndarray|float|int
        :param params: Used for overriding the parameters
        :type params: tuple of np.ndarray|float|int
        :return: The log-weights
        :rtype: np.ndarray
        """

        return self.noise0.logpdf(x, loc=self.i_mean(params), scale=self.i_scale(params))

    def mean(self, x, params=None):
        """
        Calculates the mean of the process conditional on the previous state and current parameters.
        :param x: The state of the process.
        :type x: np.ndarray|float|int
        :param params: Used for overriding the parameters
        :type params: tuple of np.ndarray|float|int
        :return: The mean
        :rtype: np.ndarray|float
        """

        return resizer(self.f(x, *(params or self.theta)))

    def scale(self, x, params=None):
        """
        Calculates the scale of the process conditional on the current state and parameters.
        :param x: The state of the process
        :type x: np.ndarray|float|int
        :param params: Used for overriding the parameters
        :type params: tuple of np.ndarray|float|int
        :return: The scale
        :rtype: np.ndarray|float
        """

        return resizer(self.g(x, *(params or self.theta)))

    def weight(self, y, x, params=None):
        """
        Weights the process of the current state `x_t` with the previous `x_{t-1}`. Used whenever the proposal
        distribution is different from the underlying.
        :param y: The value at x_t
        :type y: np.ndarray|float|int
        :param x: The value at x_{t-1}
        :type x: np.ndarray|float|int
        :param params: Used of overriding the parameters
        :type params: tuple of np.ndarray|float|int
        :return: The log-weights
        :rtype: np.ndarray|float
        """

        return self.noise.logpdf(y, loc=self.mean(x, params=params), scale=self.scale(x, params=params))

    def i_sample(self, size=None, **kwargs):
        """
        Samples from the initial distribution.
        :param size: The number of samples
        :type size: int|tuple of int
        :param kwargs: kwargs passed to the noise class
        :return: Samples from the initial distribution
        :rtype: np.ndarray|float|int
        """

        return self.noise0.rvs(loc=self.i_mean(), scale=self.i_scale(), size=size, **kwargs)

    def propagate(self, x, params=None):
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous state
        :type x: np.ndarray|float|int
        :param params: Used for overriding the parameters
        :type params: tuple of np.ndarray|float|int
        :return: Samples from the model
        :rtype: np.ndarray|float|int
        """

        loc = self.mean(x, params)
        scale = self.scale(x, params)

        return self.noise.rvs(loc=loc, scale=scale)

    def sample(self, steps, samples=None, **kwargs):
        """
        Samples a trajectory from the model.
        :param steps: The number of steps
        :type steps: int
        :param samples: Number of sample paths
        :type samples: int
        :param kwargs: kwargs passed to `i_sample` and `propagate`
        :return: An array of sampled values
        :rtype: np.ndarray
        """

        out = list()
        out.append(self.i_sample(size=samples, **kwargs))

        for i in range(1, steps):
            out.append(self.propagate(out[i-1], **kwargs))

        return np.array(out)

    def p_apply(self, func):
        """
        Applies `func` to each parameter of the model "inplace", i.e. manipulates `self.theta`.
        :param func: Function to apply, must be of the structure func(param)
        :type func: callable
        :return: Instance of self
        :rtype: Base
        """

        self.theta = self.p_map(func)

        return self

    def p_prior(self):
        """
        Calculates the prior log-likelihood of the current values.
        :return: The prior of the current parameter values
        :rtype: np.ndarray|float
        """

        return sum(self.p_map(lambda x: x[1].logpdf(x[0]), default=0))

    def p_map(self, func, default=None):
        """
        Applies the func to the parameters and returns a tuple of objects.
        :param func: The function to apply to parameters.
        :type func: callable
        :param default: What to set those parameters that aren't distributions to. If `None`, sets to the current value
        :type default: np.ndarray|float|int
        :return: Returns tuple of values
        :rtype: np.ndarray|float|int
        """

        out = tuple()

        for p, op in zip(self.theta, self._o_theta):
            if isinstance(op, Distribution):
                out += (func((p, op)),)
            else:
                out += (default if default is not None else p,)

        return out

    def p_grad(self, y, x, h=1e-3):
        """
        Calculates the gradient of the model parameters for current state `y` with the previous state `x`.
        :param y: The current state
        :type y: np.ndarray|float|int
        :param x: The previous state
        :type x: np.ndarray|float
        :param h: The step size
        :type h: float
        :return: The gradient of the model parameters
        :rtype: np.ndarray|float
        """

        out = tuple()
        for i, (p, op) in enumerate(zip(self.theta, self._o_theta)):
            if isinstance(op, Distribution):
                up, low = list(self.theta).copy(), list(self.theta).copy()
                up[i], low[i] = p + h, p - h
                out += ((self.weight(y, x, params=up) - self.weight(y, x, params=low)) / 2 / h,)
            else:
                out += (np.zeros_like(x),)

        return out