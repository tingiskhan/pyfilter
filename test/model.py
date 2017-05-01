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


class Tests(unittest.TestCase):
    linear = ts.Base((f0, g0), (f, g), (1, 1), (stats.norm, stats.norm))
    linearobs = ts.Base((f0, g0), (fo, go), (1, 0.1), (stats.norm, stats.norm))
    model = StateSpaceModel(linear, linearobs)

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

        truew = stats.norm.logpdf(y, loc=x[0])

        assert np.allclose(w, truew)

    def test_Sample(self):
        x, y = self.model.sample(50)

        assert len(x) == 50 and len(y) == 50

        fig, ax = plt.subplots(ncols=2)

        ax[0].plot(x)
        ax[1].plot(y)

        ax[0].set_title('Should look roughly the same')

        plt.show()
