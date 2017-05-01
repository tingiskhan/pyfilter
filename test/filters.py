from pyfilter.model import StateSpaceModel
import pyfilter.timeseries.meta as ts
import unittest
import scipy.stats as stats
import matplotlib.pyplot as plt
import pyfilter.filters.bootstrap as sisr


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
    linearobs = ts.Base((f0, g0), (fo, go), (1, 1), (stats.norm, stats.norm))
    model = StateSpaceModel(linear, linearobs)

    def test_InitializeFilter(self):
        filt = sisr.Bootstrap(self.model, 1000)

        filt.initialize()

        assert filt._old_x[0].shape == (1000,)

    def test_WeightFilter(self):

        x, y = self.model.sample(500)

        filt = sisr.Bootstrap(self.model, 1000).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_x) > 0

        estimates = filt.filtermeans()

        fig, ax = plt.subplots()

        ax.plot(x)
        ax.plot(estimates)

        plt.show()