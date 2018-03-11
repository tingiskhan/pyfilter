import numpy as np
from ..distributions.continuous import Distribution
from ..utils.utils import resizer


class Base(object):
    def __init__(self, initial, funcs, theta, noise, q=None):
        """
        This object is to serve as a base class for the timeseries models.
        :param initial: The functions governing the initial dynamics of the process
        :type initial: tuple of functions
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple of functions
        :param theta: The parameters governing the dynamics
        :type theta: tuple
        :param noise: The noise governing the noise process.
        :type noise: tuple of Distribution
        :param q: The correlation of the noise processes
        :type q: numpy.ndarray
        """

        self.f0, self.g0 = initial
        self.f, self.g = funcs
        self.theta = theta
        self._o_theta = theta
        self.noise0, self.noise = noise
        self.q = q

    @property
    def ndim(self):
        """
        Returns the dimension of the timeseries
        :return:
        """
        return self.noise.ndim

    @property
    def priors(self):
        """
        Returns the priors of the time series.
        :return:
        """
        return self._o_theta

    def i_mean(self):
        """
        Calculates the mean of the initial distribution.
        :return:
        """

        return resizer(self.f0(*self.theta))

    def i_scale(self):
        """
        Calculates the scale of the initial distribution.
        :return:
        """

        return resizer(self.g0(*self.theta))

    def mean(self, x, params=None):
        """
        Calculates the mean of the process conditional on the state and parameters.
        :param x: The state of the process.
        :param params: Used for overriding the parameters
        :return:
        """

        return resizer(self.f(x, *(params or self.theta)))

    def scale(self, x, params=None):
        """
        Calculates the scale of the process conditional on the state and parameters.
        :param x: The state of the process
        :param params: Used for overriding the parameters
        :return:
        """

        return resizer(self.g(x, *(params or self.theta)))

    def weight(self, y, x, params=None):
        """
        Weights the process of the current observation x_t with the previous x_{t-1}. Used whenever the proposal
        distribution is different from the underlying.
        :param y: The value at x_t
        :param x: The value at x_{t-1}
        :param params: The value of the parameters.
        :return:
        """

        return self.noise.logpdf(y, loc=self.mean(x, params=params), scale=self.scale(x, params=params))

    def i_sample(self, size=None, **kwargs):
        """
        Samples from the initial distribution.
        :param size: The number of samples
        :param kwargs: kwargs passed to the noise class
        :return:
        """

        return self.noise0.rvs(loc=self.i_mean(), scale=self.i_scale(), size=size, **kwargs)

    def propagate(self, x, **kwargs):
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous states
        :param params: If you wish to override the parameters used for sampling
        :param kwargs: kwargs passed to the noise class
        :return:
        """

        loc = self.mean(x, **kwargs)
        scale = self.scale(x, **kwargs)

        return self.noise.rvs(loc=loc, scale=scale)

    def sample(self, steps, samples=None, **kwargs):
        """
        Samples an entire trajectory from the model.
        :param steps: The number of steps
        :type steps: int
        :param samples: Number of sample paths.
        :type samples: int
        :param kwargs:
        :return:
        """

        out = list()
        out.append(self.i_sample(size=samples, **kwargs))

        for i in range(1, steps):
            out.append(self.propagate(out[i-1], **kwargs))

        return np.array(out)

    def p_apply(self, func):
        """
        Applies `func` to each parameter of the model. 
        :param func: Function to apply, must be of the structure func(param).
        :return: 
        """

        self.theta = self.p_map(func)

        return self

    def p_prior(self):
        """
        Calculates the prior log likelihood of the current values.
        :return:
        """

        return sum(self.p_map(lambda x: x[1].logpdf(x[0]), default=0))

    def p_map(self, func, default=None):
        """
        Applies the func to the parameters and returns a tuple of objects.
        :param func: The function to apply to parameters.
        :param default: What to set those parameters that aren't distributions to. If `None`, sets to the current value.
        :return:
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
        Calculates the gradient of the model for the current state y with the previous state x, using h as step size.
        :param y: The current state
        :param x: The previous state
        :param h: The step size.
        :return:
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