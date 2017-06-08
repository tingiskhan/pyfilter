import numpy as np
from math import pi
import scipy.stats as stats


def _get(x, y):
    """
    Returns x if not None, else y
    :param x:
    :param y:
    :return:
    """

    return x if x is not None else y


class Distribution(object):
    def logpdf(self, *args, **kwargs):
        """
        Implements the logarithm of the PDF.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def rvs(self, *args, **kwargs):
        """
        Samples from the distribution of interest
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def bounds(self):
        """
        Return the bounds on which the RV is defined.
        :return:
        """

        raise NotImplementedError()

    def std(self):
        """
        Returns the standard deviation of the RV.
        :return: 
        """

        raise NotImplementedError()


class Normal(Distribution):
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def logpdf(self, x, loc=None, scale=None, size=None, **kwargs):
        m, s = _get(loc, self.loc), _get(scale, self.scale)

        return stats.norm.logpdf(x, loc=m, scale=s)

    def rvs(self, loc=None, scale=None, size=None, **kwargs):
        m, s = _get(loc, self.loc), _get(scale, self.scale)

        return np.random.normal(loc=m, scale=s, size=size)

    def bounds(self):
        return -np.infty, np.infty

    def std(self):
        return stats.norm(loc=self.loc, scale=self.scale).std()


class Gamma(Distribution):
    def __init__(self, a, loc=0, scale=1):
        self.a = a
        self.loc = loc
        self.scale = scale

    def logpdf(self, x, a=None, loc=None, scale=None, size=None, **kwargs):
        a = _get(a, self.a)
        loc = _get(loc, self.loc)
        scale = _get(scale, self.scale)

        return stats.gamma.logpdf(x, a=a, loc=loc, scale=scale, size=size, **kwargs)

    def rvs(self, a=None, loc=None, scale=None, size=None, **kwargs):
        a = _get(a, self.a)
        loc = _get(loc, self.loc)
        scale = _get(scale, self.scale)

        return loc + np.random.gamma(a, scale, size=size)

    def bounds(self):
        return self.loc, np.infty

    def std(self):
        return stats.gamma(a=self.a, loc=self.loc, scale=self.scale).std()


class Beta(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def rvs(self, a=None, b=None, size=None, **kwargs):
        a, b = _get(a, self.a), _get(b, self.b)

        return np.random.beta(a, b, size=size)

    def logpdf(self, x, a=None, b=None, size=None, **kwargs):
        return stats.beta.logpdf(x, a, b)

    def bounds(self):
        return 0, 1

    def std(self):
        return stats.beta(a=self.a, b=self.b).std()


class MultivariateNormal(Distribution):
    def __init__(self, mean=np.zeros(2), cov=np.eye(2)):
        self.mean = mean
        self.cov = cov

        self._hmean = np.zeros_like(mean)

    def rvs(self, loc=None, scale=None, size=None, **kwargs):
        loc, scale = _get(loc, self.mean), _get(scale, self.cov)

        rvs = np.random.multivariate_normal(mean=self._hmean, cov=self.cov, size=size)

        return loc + np.einsum('ij...,...i->i...', scale, rvs)

    def logpdf(self, x, loc=None, scale=None, **kwargs):
        loc, scale = _get(loc, self.mean), _get(scale, self.cov)
        # TODO: Figure out a way to calculate the log pdf
        raise NotImplementedError()

    def bounds(self):
        bound = np.infty * np.ones_like(self.mean)
        return -bound, bound

    def std(self):
        return self.cov