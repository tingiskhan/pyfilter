import numpy as np
import abc
import scipy.stats as stats
import pyfilter.utils.utils as helps


def _get(x, y):
    """
    Returns x if not None, else y
    :param x:
    :param y:
    :return:
    """

    return np.array(x if x is not None else y)


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


class OneDimensional(Distribution):
    ndim = 1

    def cov(self):
        return self.std() ** 2


class MultiDimensional(Distribution):
    @abc.abstractmethod
    def ndim(self):
        return 2

    def cov(self):
        return self.std()


class Normal(OneDimensional):
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def logpdf(self, x, loc=None, scale=None, size=None, **kwargs):
        m, s = _get(loc, self.loc), _get(scale, self.scale) ** 2

        return -np.log(2 * np.pi * s) / 2 - (x - m) ** 2 / 2 / s

    def rvs(self, loc=None, scale=None, size=None, **kwargs):
        m, s = _get(loc, self.loc), _get(scale, self.scale)

        return np.random.normal(loc=m, scale=s, size=size)

    def bounds(self):
        return -np.infty, np.infty

    def std(self):
        return stats.norm(loc=self.loc, scale=self.scale).std()


class Gamma(OneDimensional):
    def __init__(self, a, loc=0, scale=1):
        self.a = a
        self.loc = loc
        self.scale = scale

    def logpdf(self, x, a=None, loc=None, scale=None, size=None, **kwargs):
        a = _get(a, self.a)
        loc = _get(loc, self.loc)
        scale = _get(scale, self.scale)

        return stats.gamma.logpdf(x, a=a, loc=loc, scale=scale, **kwargs)

    def rvs(self, a=None, loc=None, scale=None, size=None, **kwargs):
        a = _get(a, self.a)
        loc = _get(loc, self.loc)
        scale = _get(scale, self.scale)

        return loc + np.random.gamma(a, scale, size=size)

    def bounds(self):
        return self.loc, np.infty

    def std(self):
        return stats.gamma(a=self.a, loc=self.loc, scale=self.scale).std()


class InverseGamma(OneDimensional):
    def __init__(self, a, loc=0, scale=1):
        self.a = a
        self.loc = loc
        self.scale = scale

    def std(self):
        return stats.invgamma(a=self.a, loc=self.loc, scale=self.scale).std()

    def logpdf(self, x, a=None, loc=None, scale=None, size=None, **kwargs):
        a = _get(a, self.a)
        loc = _get(loc, self.loc)
        scale = _get(scale, self.scale)

        return stats.invgamma.logpdf(x, a=a, loc=loc, scale=scale, **kwargs)

    def rvs(self, a=None, loc=None, scale=None, size=None, **kwargs):
        a = _get(a, self.a)
        loc = _get(loc, self.loc)
        scale = _get(scale, self.scale)

        return stats.invgamma(a, scale=scale, loc=loc).rvs(size=size)

    def bounds(self):
        return self.loc, np.infty


class Beta(OneDimensional):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def rvs(self, a=None, b=None, size=None, **kwargs):
        a, b = _get(a, self.a), _get(b, self.b)

        return np.random.beta(a, b, size=size)

    def logpdf(self, x, a=None, b=None, size=None, **kwargs):
        a, b = _get(a, self.a), _get(b, self.b)
        return stats.beta.logpdf(x, a, b)

    def bounds(self):
        return 0, 1

    def std(self):
        return stats.beta(a=self.a, b=self.b).std()


class MultivariateNormal(MultiDimensional):
    def __init__(self, mean=np.zeros(2), scale=np.eye(2), ndim=None):

        if ndim:
            self.mean = np.zeros(ndim)
            self.cov = np.eye(ndim)
            self._ndim = ndim
        else:
            self.mean = mean
            self.cov = scale
            self._ndim = scale.shape[0]

        self._hmean = np.zeros(self._ndim)
        self._hcov = np.eye(self._ndim)

    @property
    def ndim(self):
        return self._ndim

    def rvs(self, loc=None, scale=None, size=None, **kwargs):
        loc, scale = _get(loc, self.mean), _get(scale, self.cov)

        rvs = np.random.multivariate_normal(mean=self._hmean, cov=self._hcov, size=(size or loc.shape[1:]))

        return (loc.T + np.einsum('ij...,...i->i...', scale, rvs).T).T

    def logpdf(self, x, loc=None, scale=None, **kwargs):
        loc, scale = _get(loc, self.mean), _get(scale, np.eye(self.cov.shape[0]))

        cov = helps.outer(scale, self.cov)

        t1 = - 0.5 * np.log((2 * np.pi) ** self._ndim * np.linalg.det(cov.T)).T
        t2 = - helps.square((x.T - loc.T).T, cov)

        return t1 + t2

    def bounds(self):
        bound = np.infty * np.ones_like(self.mean)
        return -bound, bound

    def std(self):
        return self.cov