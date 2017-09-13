from pyfilter.model import StateSpaceModel
import pyfilter.timeseries.meta as ts
from pyfilter.timeseries.observable import Observable
import unittest
import scipy.stats as stats
import pyfilter.filters as sisr
import pyfilter.filters as apf
from pyfilter.filters import RAPF
from pyfilter.filters import NESS
from pyfilter.filters import SMC2
from pyfilter.utils.normalization import normalize
from pyfilter.filters import Linearized
import pykalman
import numpy as np
from pyfilter.distributions.continuous import Normal, Gamma
from pyfilter.proposals import Linearized as Linz


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
        filt = sisr.SISR(self.model, 1000)

        filt.initialize()

        assert filt._old_x[0].shape == (1000,)

    def test_WeightFilter(self):

        x, y = self.model.sample(500)

        filt = sisr.SISR(self.model, 5000, saveall=True).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_x) > 0

        estimates = np.array(filt.filtermeans())

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0]) ** 2))

        assert rmse < 0.05

    def test_APF(self):
        x, y = self.model.sample(500)

        filt = apf.APF(self.model, 5000, saveall=True).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_x) > 0

        estimates = np.array(filt.filtermeans())

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0]) ** 2))

        assert rmse < 0.05

    def test_Likelihood(self):
        x, y = self.model.sample(500)

        apft = apf.APF(self.model, 1000, proposal=Linz).initialize().longfilter(y)
        sisrt = sisr.SISR(self.model, 1000).initialize().longfilter(y)
        linearizedt = Linearized(self.model, 1000).initialize().longfilter(y)

        rmse = np.sqrt(np.mean((np.array(apft.s_l) - np.array(sisrt.s_l)) ** 2))
        rmse2 = np.sqrt(np.mean((np.array(linearizedt.s_l) - np.array(sisrt.s_l)) ** 2))

        assert (rmse < 0.1) and (rmse2 < 0.1)

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        kalmanloglikelihood = kf.loglikelihood(y)

        apferror = np.abs((kalmanloglikelihood - np.array(apft.s_l).sum()) / kalmanloglikelihood)
        sisrerror = np.abs((kalmanloglikelihood - np.array(sisrt.s_l).sum()) / kalmanloglikelihood)
        linerror = np.abs((kalmanloglikelihood - np.array(linearizedt.s_l).sum()) / kalmanloglikelihood)

        assert (apferror < 0.01) and (sisrerror < 0.01) and (linerror < 0.01)

    def test_MultiDimensional(self):
        x, y = self.model.sample(50)

        shape = 50, 1

        linear = ts.Base((f0, g0), (f, g), (np.ones(shape), np.ones(shape)), (stats.norm, stats.norm))
        self.model.hidden = (linear,)

        apft = apf.APF(self.model, (shape[0], 1000)).initialize().longfilter(y)

        filtermeans = np.array(apft.filtermeans())

        rmse = np.sqrt(np.mean((filtermeans[:, 0:1] - filtermeans[:, 1:]) ** 2))

        assert rmse < 0.1

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

    def test_SMC2(self):
        x, y = self.model.sample(300)

        linear = ts.Base((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = (linear,)
        self.model.observable = ts.Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        smc2 = SMC2(self.model, (300, 300))

        smc2 = smc2.longfilter(y)

        weights = normalize(smc2._recw)

        mean = np.average(smc2._filter._model.hidden[0].theta[1], weights=weights[:, None])
        std = np.sqrt(np.average((smc2._filter._model.hidden[0].theta[1] - mean) ** 2, weights=weights[:, None]))

        assert mean - std < 1 < mean + std

    def test_Linearized(self):
        x, y = self.model.sample(500)

        filt = Linearized(self.model, 5000, saveall=True).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_x) > 0

        estimates = np.array(filt.filtermeans())

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0]) ** 2))

        assert rmse < 0.05

    def test_Gradient(self):
        x, y = self.model.sample(500)

        linear = ts.Base((f0, g0), (f, g), (1., Gamma(1)), (Normal(), Normal()))

        self.model.hidden = (linear,)
        self.model.observable = ts.Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))

        rapf = RAPF(self.model, 3000).initialize().longfilter(y)

        grad = self.model.p_grad(y[-1], rapf.s_x[-1], rapf.s_x[-2])

        def truderiv(obs, mu, sigma):
            return ((obs - mu) ** 2 - sigma ** 2) / sigma ** 3

        truederiv = truderiv(y[-1], rapf.s_x[-1][0], self.model.observable.theta[-1])

        assert np.allclose(truederiv, grad[-1][-1], atol=1e-4)
