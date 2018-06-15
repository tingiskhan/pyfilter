import numpy as np
from scipy.special import logit, expit


class TransformMixin(object):
    def transform(self, x):
        """
        Implements the transformation that maps from variable definition space to the real line. Heavily inspired by
        PyMC3's, but cannot use theirs as they don't have a backward_val.
        :param x: The value(s) to transform
        :type x: np.ndarray|float
        :return: A new array
        :rtype: np.ndarray
        """

        raise NotImplementedError()

    def inverse_transform(self, x):
        """
        Implements the transformation that maps from real line to variable definition space.
        :param x: The value(s) to transform
        :type x: np.ndarray|float
        :return: A new array
        :rtype: np.ndarray
        """

        raise NotImplementedError()


class NonTransformable(TransformMixin):
    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class LogOdds(TransformMixin):
    """
    Implements transform for variables defined [0, 1].
    """

    def inverse_transform(self, x):
        return expit(x)

    def transform(self, x):
        return logit(x)


class Log(TransformMixin):
    """
    Implements a transformation for strictly positive variables, i.e. > 0.
    """

    def transform(self, x):
        return np.log(x)

    def inverse_transform(self, x):
        return np.exp(x)


class Interval(TransformMixin):
    """
    Implements the transform for variables defined [a, b].
    """

    def transform(self, x):
        a, b = self.bounds()

        return (b - a) * expit(x) + a

    def inverse_transform(self, x):
        a, b = self.bounds()

        return np.log(x - a) - np.log(b - x)