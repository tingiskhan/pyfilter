import unittest
import pyfilter.timeseries.meta as ts
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


class Tests(unittest.TestCase):
    linear = ts.Base((f0, g0), (f, g), (1, 1), (stats.norm, stats.norm))

    def test_TimeseriesCreate_1D(self):

        assert self.linear.mean(1) == 1 and self.linear.scale(1) == 1

    def test_Timeseries2DState(self):
        x = np.random.normal(size=500)

        assert np.allclose(self.linear.mean(x), f(x, 1, 1))

    def test_Timeseries3DState2DParam(self):
        alpha = 0.5 * np.ones((500, 1))
        x = np.random.normal(size=(500, 200))
        linear = ts.Base((f0, g0), (f, g), (alpha, 1), (stats.norm, stats.norm))

        assert np.allclose(linear.mean(x), f(x, alpha, 1))

    def test_SampleInitial(self):
        x = self.linear.i_sample()

        assert x is not None

    def test_Propagate(self):
        out = np.zeros(500)

        out[0] = self.linear.i_sample()

        for i in range(1, out.shape[0]):
            out[i] = self.linear.propagate(out[i-1])

        plt.plot(out)
        plt.show()

        assert True

    def test_SampleTrajectory1D(self):
        sample = self.linear.sample(500)

        assert sample.shape[0] == 500

    def test_SampleTrajectory2D(self):
        sample = self.linear.sample(500, 250)

        assert sample.shape == (500, 250)