from pyfilter.model import StateSpaceModel
import pyfilter.timeseries.meta as ts
import unittest
import scipy.stats as stats
import pyfilter.filters.bootstrap as sisr
import pyfilter.filters.apf as apf
from pyfilter.filters.rapf import RAPF
import pykalman
import numpy as np
import matplotlib.pyplot as plt
from pyfilter.distributions.continuous import Normal, Gamma


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
    linear = ts.Base((f0, g0), (f, g), (1, 1), (Normal(), Normal()))
    linearobs = ts.Base((f0, g0), (fo, go), (1, 1), (Normal(), Normal()))
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

    def test_APF(self):
        x, y = self.model.sample(500)

        filt = apf.APF(self.model, 5000).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_x) > 0

        estimates = filt.filtermeans()

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        maxerr = np.abs(estimates - filterestimates[0]).max()

        assert maxerr < 0.5

    def test_Likelihood(self):
        x, y = self.model.sample(500)

        apft = apf.APF(self.model, 1000).initialize().longfilter(y)
        sisrt = sisr.Bootstrap(self.model, 1000).initialize().longfilter(y)

        fig, ax = plt.subplots()

        ax.plot(apft.s_l, label='APF')
        ax.plot(sisrt.s_l, label='SISR')

        ax.set_title('Verify that APF > SISR in general')
        plt.show()

    def test_MultiDimensional(self):
        x, y = self.model.sample(50)

        shape = 50, 1

        linear = ts.Base((f0, g0), (f, g), (np.ones(shape), np.ones(shape)), (stats.norm, stats.norm))
        self.model.hidden = (linear,)

        apft = apf.APF(self.model, (shape[0], 1000)).initialize().longfilter(y)

        filtermeans = apft.filtermeans()

        for i in range(shape[0]):
            plt.plot(filtermeans[:, :, i])

        plt.show()

    def test_RAPFSimpleModel(self):
        x, y = self.model.sample(500)

        linear = ts.Base((f0, g0), (f, g), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))

        self.model.hidden = (linear,)
        self.model.observable = ts.Base((f0, g0), (fo, go), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))
        rapf = RAPF(self.model, 5000).initialize()

        assert rapf._model.hidden[0].theta[1].shape == (5000,)

        rapf = rapf.longfilter(y)

        estimates = rapf._model.hidden[0].theta[1]

        mean = np.mean(estimates)
        std = np.std(estimates)

        assert mean - 3 * std < 1 < mean + 3 * std

