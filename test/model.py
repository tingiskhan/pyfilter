from pyfilter.model import StateSpaceModel
import pyfilter.timeseries.meta as ts
import unittest
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


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


def foo(x, alpha, sigma):
    return alpha * x[0] + x[1]


def goo(x, alpha, sigma):
    return sigma


class Tests(unittest.TestCase):
    linear = ts.Base((f0, g0), (f, g), (1, 1), (stats.norm, stats.norm))
    linear2 = ts.Base((f0, g0), (f, g), (1, 1), (stats.norm, stats.norm))

    linearobs = ts.Base((f0, g0), (fo, go), (1, 0.1), (stats.norm, stats.norm))
    linearobs2 = ts.Base((f0, g0), (foo, goo), (1, 0.1), (stats.norm, stats.norm))

    model = StateSpaceModel(linear, linearobs)

    bivariatemodel = StateSpaceModel((linear, linear2), linearobs2)

    def test_InitializeModel1D(self):
        sample = self.model.initialize()

        assert isinstance(sample[0], float)

    def test_InitializeModel(self):
        sample = self.model.initialize(1000)

        assert sample[0].shape == (1000,)

    def test_Propagate(self):
        x = self.model.initialize(1000)

        sample = self.model.propagate(x)

        assert sample[0].shape == (1000,)

    def test_Weight(self):
        x = self.model.initialize(1000)

        y = 0

        w = self.model.weight(y, x)

        truew = stats.norm.logpdf(y, loc=x[0], scale=self.model.observable.theta[1])

        assert np.allclose(w, truew)

    def test_Sample(self):
        x, y = self.model.sample(50)

        assert len(x) == 50 and len(y) == 50

        fig, ax = plt.subplots(ncols=2)

        ax[0].plot(x)
        ax[1].plot(y)

        ax[0].set_title('Should look roughly the same')

        plt.show()

    def test_SampleBivariate(self):
        x, y = self.bivariatemodel.sample(50)

        assert len(x) == 50 and len(y) == 50

        fig, ax = plt.subplots(ncols=2)

        ax[0].plot(x)
        ax[1].plot(y)

        ax[0].set_title('Make sure there are two different lines on the left and only one on the right')

        plt.show()