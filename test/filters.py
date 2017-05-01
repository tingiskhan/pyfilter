from pyfilter.model import StateSpaceModel
import pyfilter.timeseries.meta as ts
import unittest
import scipy.stats as stats
import pyfilter.filters.bootstrap as sisr
import pykalman
import numpy as np


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

        filt = sisr.Bootstrap(self.model, 5000).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_x) > 0

        estimates = filt.filtermeans()

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        maxerr = np.abs(estimates - filterestimates[0]).max()

        assert maxerr < 0.5
