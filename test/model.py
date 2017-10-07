import unittest

import numpy as np
import scipy.stats as stats

import pyfilter.distributions.continuous as cont
import pyfilter.utils.utils as helps
from pyfilter.timeseries import StateSpaceModel, Observable, Base


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return 0


def g0(alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x


def go(x, alpha, sigma):
    return sigma


def foo(x1, x2, alpha, sigma):
    return alpha * x1 + x2


def goo(x1, x2, alpha, sigma):
    return sigma


def fmvn(x, a, sigma):
    return helps.dot(a, x)


def f0mvn(a, sigma):
    return [0, 0]


def fomvn(x, sigma):
    return x[0] + x[1] / 2


def gomvn(x, sigma):
    return sigma


class Tests(unittest.TestCase):
    linear = Base((f0, g0), (f, g), (1, 1), (stats.norm, stats.norm))

    linearobs = Observable((fo, go), (1, 1), stats.norm)

    model = StateSpaceModel(linear, linearobs)

    mat = np.eye(2)
    scale = np.eye(2)

    mvnlinear = Base((f0mvn, g0), (fmvn, g), (mat, scale), (cont.MultivariateNormal(), cont.MultivariateNormal()))
    mvnoblinear = Observable((fomvn, gomvn), (1,), cont.Normal())

    mvnmodel = StateSpaceModel(mvnlinear, mvnoblinear)

    def test_InitializeModel1D(self):
        sample = self.model.initialize()

        assert isinstance(sample, float)

    def test_InitializeModel(self):
        sample = self.model.initialize(1000)

        assert sample.shape == (1000,)

    def test_Propagate(self):
        x = self.model.initialize(1000)

        sample = self.model.propagate(x)

        assert sample.shape == (1000,)

    def test_Weight(self):
        x = self.model.initialize(1000)

        y = 0

        w = self.model.weight(y, x)

        truew = stats.norm.logpdf(y, loc=x, scale=self.model.observable.theta[1])

        assert np.allclose(w, truew)

    def test_Sample(self):
        x, y = self.model.sample(50)

        assert len(x) == 50 and len(y) == 50 and np.array(x).shape == (50,)

    def test_SampleMultivariate(self):
        x, y = self.mvnmodel.sample(30)

        assert len(x) == 30 and x[0].shape == (2,)