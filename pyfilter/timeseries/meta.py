import numpy as np


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
        :type noise: tuple
        :param q: The correlation of the noise processes
        :type q: numpy.ndarray
        """

        self.f0, self.g0 = initial
        self.f, self.g = funcs
        self.theta = theta
        self.noise0, self.noise = noise
        self.q = q

    def i_mean(self):
        """
        Calculates the mean of the initial distribution.
        :return:
        """

        return self.f0(*self.theta)

    def i_scale(self):
        """
        Calculates the scale of the initial distribution.
        :return:
        """

        return self.g0(*self.theta)

    def mean(self, x, *args, params=None):
        """
        Calculates the mean of the process conditional on the state and parameters.
        :param x: The state of the process.
        :param params: Used for overriding the parameters
        :return:
        """

        return self.f(x, *args, *(params or self.theta))

    def scale(self, x, *args, params=None):
        """
        Calculates the scale of the process conditional on the state and parameters.
        :param x: The state of the process
        :param params: Used for overriding the parameters
        :return:
        """

        return self.g(x, *args, *(params or self.theta))

    def weight(self, y, x, *args):
        """
        Weights the process of the current observation x_t with the previous x_{t-1}. Used whenever the proposal
        distribution is different from the underlying.
        :param y: The value at x_t
        :param x: The value at x_{t-1}
        :return:
        """

        return self.noise.logpdf(y, loc=self.mean(x, *args), scale=self.scale(x, *args))

    def i_sample(self, size=None, **kwargs):
        """
        Samples from the initial distribution.
        :param size: The number of samples
        :param kwargs: kwargs passed to the noise class
        :return:
        """

        return self.noise0.rvs(loc=self.i_mean(), scale=self.i_scale(), size=size, **kwargs)

    def propagate(self, x, *args, params=None, **kwargs):
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous states
        :param params: If you wish to override the parameters used for sampling
        :param kwargs: kwargs passed to the noise class
        :return:
        """

        loc = self.mean(x, *args, params=params)
        scale = self.scale(x, *args, params=params)

        return self.noise.rvs(loc=loc, scale=scale, **kwargs)

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