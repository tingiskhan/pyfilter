import unittest
import numpy as np
import pykalman
import scipy.stats as stats
from pyfilter.distributions.continuous import Normal, Gamma, MultivariateNormal
from pyfilter.filters import Linearized, NESS, RAPF, SMC2, SISR, APF, UPF, GlobalUPF, UKF
from pyfilter.proposals import Linearized as Linz, Unscented
from pyfilter.timeseries import StateSpaceModel, Observable, Base
from pyfilter.utils.normalization import normalize
from pyfilter.utils.utils import dot


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


def fmvn(x, alpha, sigma):
    return dot(np.array([[alpha, 1 / 3], [0, 1]]), x)


def gmvn(x, alpha, sigma):
    return [[sigma, 0], [0, sigma]]


def f0mvn(alpha, sigma):
    return [0, 0]


def g0mvn(alpha, sigma):
    return [[sigma, 0], [0, sigma]]


def fomvn(x, alpha, sigma):
    return x[0] + 2 * x[1]


class Tests(unittest.TestCase):
    linear = Base((f0, g0), (f, g), (1, 1), (Normal(), Normal()))
    linearobs = Observable((fo, go), (1, 1), Normal())
    model = StateSpaceModel(linear, linearobs)

    mvn = Base((f0mvn, g0mvn), (fmvn, gmvn), (0.5, 1), (MultivariateNormal(), MultivariateNormal()))
    mvnobs = Observable((fomvn, go), (1, 1), Normal())
    mvnmodel = StateSpaceModel(mvn, mvnobs)

    def test_InitializeFilter(self):
        filt = SISR(self.model, 1000)

        filt.initialize()

        assert filt._old_x.shape == (1000,)

    def test_SISR(self):

        x, y = self.model.sample(500)

        filt = SISR(self.model, 5000, saveall=True).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_mx) > 0

        estimates = np.array(filt.filtermeans())

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0][:, 0]) ** 2))

        assert rmse < 0.05

    def test_APF(self):
        x, y = self.model.sample(500)

        filt = APF(self.model, 5000, saveall=True).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_mx) > 0

        estimates = np.array(filt.filtermeans())

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0][:, 0]) ** 2))

        assert rmse < 0.05

    def test_Likelihood(self):
        x, y = self.model.sample(500)

        apft = APF(self.model, 1000).initialize().longfilter(y)
        sisrt = SISR(self.model, 1000).initialize().longfilter(y)
        linearizedt = Linearized(self.model, 1000).initialize().longfilter(y)
        upf = UPF(self.model, 1000).initialize().longfilter(y)
        ukf = UKF(self.model).initialize().longfilter(y)

        rmse = np.sqrt(np.mean((np.array(apft.s_l) - np.array(sisrt.s_l)) ** 2))
        rmse2 = np.sqrt(np.mean((np.array(linearizedt.s_l) - np.array(sisrt.s_l)) ** 2))
        rmse3 = np.sqrt(np.mean((np.array(upf.s_l) - np.array(sisrt.s_l)) ** 2))
        rmse4 = np.sqrt(np.mean((np.array(ukf.s_l) - np.array(sisrt.s_l)) ** 2))

        assert (rmse < 0.1) and (rmse2 < 0.1) and (rmse3 < 0.1) and (rmse4 < 0.1)

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        kalmanloglikelihood = kf.loglikelihood(y)

        apferror = np.abs((kalmanloglikelihood - np.array(apft.s_l).sum()) / kalmanloglikelihood)
        sisrerror = np.abs((kalmanloglikelihood - np.array(sisrt.s_l).sum()) / kalmanloglikelihood)
        linerror = np.abs((kalmanloglikelihood - np.array(linearizedt.s_l).sum()) / kalmanloglikelihood)
        upferror = np.abs((kalmanloglikelihood - np.array(upf.s_l).sum()) / kalmanloglikelihood)
        ukferr = np.abs((kalmanloglikelihood - np.array(ukf.s_l).sum()) / kalmanloglikelihood)

        assert (apferror < 0.01) and (sisrerror < 0.01) and (linerror < 0.01) and (upferror < 0.01) and (ukferr < 0.01)

    def test_MultiDimensional(self):
        x, y = self.model.sample(50)

        shape = 50, 1

        linear = Base((f0, g0), (f, g), (np.ones(shape), np.ones(shape)), (stats.norm, stats.norm))
        self.model.hidden = linear

        apft = APF(self.model, (shape[0], 1000)).initialize().longfilter(y)

        filtermeans = np.array(apft.filtermeans())

        rmse = np.sqrt(np.mean((filtermeans[:, 0:1] - filtermeans[:, 1:]) ** 2))

        assert rmse < 0.1

    def test_RAPFSimpleModel(self):
        x, y = self.model.sample(500)

        linear = Base((f0, g0), (f, g), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = Base((f0, g0), (fo, go), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))
        rapf = RAPF(self.model, 5000).initialize()

        assert rapf._model.hidden.theta[1].shape == (5000,)

        rapf = rapf.longfilter(y)

        estimates = rapf._model.hidden.theta[1]

        mean = np.mean(estimates)
        std = np.std(estimates)

        assert mean - 3 * std < 1 < mean + 3 * std

    def test_Predict(self):
        x, y = self.model.sample(550)

        linear = Base((f0, g0), (f, g), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = Base((f0, g0), (fo, go), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))
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

        linear = Base((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        ness = NESS(self.model, (300, 300))

        ness = ness.longfilter(y[:500])

        estimates = ness._filter._model.hidden.theta[1]

        mean = np.mean(estimates)
        std = np.std(estimates)

        assert mean - std < 1 < mean + std

    def test_NESSPredict(self):
        x, y = self.model.sample(550)

        linear = Base((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        ness = NESS(self.model, (300, 300))

        ness = ness.longfilter(y[:500])

        x_pred, y_pred = ness.predict(50)

        for i in range(len(y_pred)):
            lower = np.percentile(y_pred[i], 1)
            upper = np.percentile(y_pred[i], 99)

            assert (y[500 + i] >= lower) and (y[500 + i] <= upper)

    def test_SMC2(self):
        x, y = self.model.sample(300)

        linear = Base((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        smc2 = SMC2(self.model, (300, 300))

        smc2 = smc2.longfilter(y)

        weights = normalize(smc2._recw)

        mean = np.average(smc2._filter._model.hidden.theta[1], weights=weights[:, None])
        std = np.sqrt(np.average((smc2._filter._model.hidden.theta[1] - mean) ** 2, weights=weights[:, None]))

        assert mean - std < 1 < mean + std

    def test_Linearized(self):
        x, y = self.model.sample(500)

        filt = Linearized(self.model, 750, saveall=True).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_mx) > 0

        estimates = np.array(filt.filtermeans())

        kf = pykalman.KalmanFilter(transition_matrices=1, observation_matrices=1)
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0][:, 0]) ** 2))

        assert rmse < 0.05

    def test_Gradient(self):
        x, y = self.model.sample(500)

        linear = Base((f0, g0), (f, g), (1., Gamma(1)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = Base((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))

        rapf = RAPF(self.model, 3000).initialize().longfilter(y)

        grad = self.model.p_grad(y[-1], rapf.s_mx[-1], rapf.s_mx[-2])

        def truderiv(obs, mu, sigma):
            return ((obs - mu) ** 2 - sigma ** 2) / sigma ** 3

        truederiv = truderiv(y[-1], rapf.s_mx[-1], self.model.observable.theta[-1])

        assert np.allclose(truederiv, grad[-1][-1], atol=1e-4)

    def test_Linearized2D(self):
        x, y = self.mvnmodel.sample(500)

        filt = Linearized(self.mvnmodel, 5000, saveall=True).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_mx) > 0

        estimates = np.array(filt.filtermeans())

        kf = pykalman.KalmanFilter(transition_matrices=[[0.5, 1/3], [0, 1]], observation_matrices=[1, 2])
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0]) ** 2))

        assert rmse < 0.05

    def test_Unscented2D(self):
        x, y = self.mvnmodel.sample(500)

        for filtr in [UPF, GlobalUPF]:
            filt = filtr(self.mvnmodel, 3000).initialize()

            filt = filt.longfilter(y)

            assert len(filt.s_mx) > 0

            estimates = np.array(filt.filtermeans())

            kf = pykalman.KalmanFilter(transition_matrices=[[0.5, 1/3], [0, 1]], observation_matrices=[1, 2])
            filterestimates = kf.filter(y)

            rmse = np.sqrt(np.mean((estimates - filterestimates[0]) ** 2))

            assert rmse < 0.05

    def test_UKF(self):
        x, y = self.mvnmodel.sample(500)

        filt = UKF(self.mvnmodel).initialize()

        filt = filt.longfilter(y)

        assert len(filt.s_mx) > 0

        estimates = np.array(filt.filtermeans())

        kf = pykalman.KalmanFilter(transition_matrices=[[0.5, 1 / 3], [0, 1]], observation_matrices=[1, 2])
        filterestimates = kf.filter(y)

        rmse = np.sqrt(np.mean((estimates - filterestimates[0]) ** 2))

        assert rmse < 0.05