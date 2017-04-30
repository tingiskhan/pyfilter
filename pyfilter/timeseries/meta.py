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

    def mean(self, x):
        """
        Calculates the mean of the process conditional on the state and parameters.
        :param x: The state of the process.
        :return:
        """

        return self.f(x, *self.theta)

    def scale(self, x):
        """
        Calculates the scale of the process conditional on the state and parameters.
        :param x: The state of the process
        :return:
        """

        return self.g(x, *self.theta)

    def weight(self, y, x):
        """
        Weights the process of the current observation x_t with the previous x_{t-1}. Used whenever the proposal
        distribution is different from the underlying.
        :param y: The value at x_t
        :param x: The value at x_{t-1}
        :return:
        """

        return self.noise(y, loc=self.mean(x), scale=self.scale(x))