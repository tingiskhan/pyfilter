from pyfilter.model import StateSpaceModel
import pyfilter.timeseries.meta as ts
from pyfilter.timeseries.observable import Observable
import unittest
import scipy.stats as stats
import pyfilter.filters as sisr
import pyfilter.filters as apf
from pyfilter.filters import RAPF
from pyfilter.filters import NESS
from pyfilter.filters.upf import UPF
from pyfilter.filters import SMC2
from pyfilter.helpers.normalization import normalize
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
    linearobs = Observable((fo, go), (1, 1), Normal())
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

        rmse = np.sqrt(np.mean((estimates - filterestimates[0][:, 0]) ** 2))

        assert rmse < 0.05

    def test_APF(self):
        x, y = self.model.sample(500)

        filt = apf.APF(self.model, 5000).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_x) > 0

        estimates = filt.filtermeans()

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0][:, 0]) ** 2))

        assert rmse < 0.05

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

    def test_Predict(self):
        x, y = self.model.sample(550)

        linear = ts.Base((f0, g0), (f, g), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))

        self.model.hidden = (linear,)
        self.model.observable = ts.Base((f0, g0), (fo, go), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))
        rapf = RAPF(self.model, 5000).initialize()

        assert rapf._model.hidden[0].theta[1].shape == (5000,)

        rapf = rapf.longfilter(y[:500])

        x_pred, y_pred = rapf.predict(50)

        for i in range(len(y_pred)):
            lower = np.percentile(y_pred[i], 1)
            upper = np.percentile(y_pred[i], 99)

            assert (y[500 + i] >= lower) and (y[500 + i] <= upper)

    def test_NESS(self):
        x, y = self.model.sample(500)

        linear = ts.Base((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = (linear,)
        self.model.observable = ts.Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        ness = NESS(self.model, (300, 300))

        ness = ness.longfilter(y[:500])

        estimates = ness._filter._model.hidden[0].theta[1]

        mean = np.mean(estimates)
        std = np.std(estimates)

        assert mean - std < 1 < mean + std

    def test_NESSPredict(self):
        x, y = self.model.sample(550)

        linear = ts.Base((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = (linear,)
        self.model.observable = ts.Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        ness = NESS(self.model, (300, 300))

        ness = ness.longfilter(y[:500])

        x_pred, y_pred = ness.predict(50)

        for i in range(len(y_pred)):
            lower = np.percentile(y_pred[i], 1)
            upper = np.percentile(y_pred[i], 99)

            assert (y[500 + i] >= lower) and (y[500 + i] <= upper)

    def test_UPF(self):
        upf = UPF(self.model, 500).initialize()

        assert upf._extendedmean.shape == (3,) and upf._extendedcov.shape == (3, 3)

        upfmd = UPF(self.model, (500, 300)).initialize()

        assert upfmd._extendedmean.shape == (3, 500) and upfmd._extendedcov.shape == (3, 3, 500)

    def test_SMC2(self):
        x, y = self.model.sample(500)

        linear = ts.Base((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = (linear,)
        self.model.observable = ts.Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        smc2 = SMC2(self.model, (3000, 100), filt=apf.APF)

        smc2 = smc2.longfilter(y)

        weights = normalize(smc2._recw)

        mean = np.average(smc2._filter._model.hidden[0].theta[1], weights=weights[:, None])
        std = np.sqrt(np.average((smc2._filter._model.hidden[0].theta[1] - mean) ** 2, weights=weights[:, None]))

        assert mean - std < 1 < mean + std