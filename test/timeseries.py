import unittest
import pyfilter.timeseries.meta as ts
import scipy.stats as stats
import numpy as np


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return 0


def g0(alpha, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_TimeseriesCreate_1D(self):
        linear = ts.Base((f0, g0), (f, g), (0.5, 1), (stats.norm, stats.norm))

        assert linear.mean(1) == 0.5 and linear.scale(1) == 1

    def test_Timeseries2DState(self):
        x = np.random.normal(size=500)
        linear = ts.Base((f0, g0), (f, g), (0.5, 1), (stats.norm, stats.norm))

        assert np.allclose(linear.mean(x), f(x, 0.5, 1))

    def test_Timeseries3DState2DParam(self):
        alpha = 0.5 * np.ones((500, 1))
        x = np.random.normal(size=(500, 200))
        linear = ts.Base((f0, g0), (f, g), (alpha, 1), (stats.norm, stats.norm))

        assert np.allclose(linear.mean(x), f(x, alpha, 1))