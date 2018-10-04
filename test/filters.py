import unittest
import numpy as np
import pykalman
import scipy.stats as stats
from torch.distributions import Normal, MultivariateNormal
from pyfilter.filters import Linearized, RAPF, SISR, APF, UPF, GlobalUPF, UKF, KalmanLaplace
from pyfilter.timeseries import StateSpaceModel, Observable, BaseModel
from pyfilter.utils.normalization import normalize
import torch


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
    x1 = alpha * x[0] + x[1] / 3
    x2 = x[1]
    return torch.cat((x1[None], x2[None]))


def gmvn(x, alpha, sigma):
    return sigma


def f0mvn(alpha, sigma):
    return torch.zeros(2)


def g0mvn(alpha, sigma):
    return sigma


def fomvn(x, alpha, sigma):
    return x[0] + 2 * x[1]


class Tests(unittest.TestCase):
    # ===== Simple 1D model ===== #
    norm = Normal(0., 1.)
    linear = BaseModel((f0, g0), (f, g), (1., 1.), (norm, norm))
    linearobs = Observable((fo, go), (1., 1.), norm)
    model = StateSpaceModel(linear, linearobs)

    # ===== Simple 2D model ===== #
    mvn = MultivariateNormal(torch.zeros(2), scale_tril=torch.eye(2))
    mvn = BaseModel((f0mvn, g0mvn), (fmvn, gmvn), (0.5, 1.), (mvn, mvn))
    mvnobs = Observable((fomvn, go), (1., 1.), norm)
    mvnmodel = StateSpaceModel(mvn, mvnobs)

    def test_InitializeFilter(self):
        filt = SISR(self.model, 1000).initialize()

        assert filt._x_cur.shape == (1000,)

    def test_Filters(self):
        for model in [self.mvnmodel]:
            x, y = model.sample(500)

            for filter_, props in [(SISR, {'particles': 5000}), (APF, {'particles': 5000})]:
                filt = filter_(model, **props).initialize()

                filt = filt.longfilter(y)

                assert len(filt.s_mx) > 0

                filtmeans = filt.filtermeans()

                # ===== Run Kalman ===== #
                if model is self.model:
                    kf = pykalman.KalmanFilter(transition_matrices=1., observation_matrices=1.)
                    estimates = np.array(filtmeans)
                else:
                    kf = pykalman.KalmanFilter(transition_matrices=[[0.5, 1 / 3], [0, 1.]], observation_matrices=[1, 2])
                    estimates = torch.cat(filtmeans, 0).numpy().reshape(-1, 2)

                filterestimates = kf.filter(y.numpy())

                if estimates.ndim < 2:
                    estimates = estimates[:, None]

                rel_error = np.abs((estimates - filterestimates[0]) / filterestimates[0]).mean()

                ll = kf.loglikelihood(y.numpy())

                rel_ll_error = np.abs((ll - np.array(filt.s_ll).sum()) / ll)

                assert rel_error < 0.05 and rel_ll_error < 0.05

    def test_MultiDimensional(self):
        x, y = self.model.sample(50)

        shape = 50, 1

        linear = BaseModel((f0, g0), (f, g), (0.5, 1.), (self.norm, self.norm))
        self.model.hidden = linear

        filt = SISR(self.model, (shape[0], 1000)).initialize().longfilter(y)

        filtermeans = np.array(filt.filtermeans())

        rmse = np.sqrt(np.mean((filtermeans[:, 0:1] - filtermeans[:, 1:]) ** 2))

        assert rmse < 0.1

    def test_RAPFSimpleModel(self):
        x, y = self.model.sample(500)

        linear = BaseModel((f0, g0), (f, g), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = BaseModel((f0, g0), (fo, go), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))
        rapf = RAPF(self.model, 5000).initialize()

        assert rapf._model.hidden.theta[1].values.shape == (5000,)

        rapf = rapf.longfilter(y)

        estimates = rapf._model.hidden.theta[1]

        mean = np.mean(estimates.values)
        std = np.std(estimates.values)

        assert mean - std < 1 < mean + std

    def test_Predict(self):
        x, y = self.model.sample(550)

        linear = BaseModel((f0, g0), (f, g), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = BaseModel((f0, g0), (fo, go), (1, Gamma(a=1, scale=2)), (Normal(), Normal()))
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

        linear = BaseModel((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = BaseModel((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        ness = NESS(self.model, (3000,), filt=UKF)

        ness = ness.longfilter(y)

        estimates = ness._filter._model.hidden.theta[1]

        mean = np.mean(estimates.values)
        std = np.std(estimates.values)

        assert mean - std < 1 < mean + std

    def test_NESSPredict(self):
        x, y = self.model.sample(550)

        linear = BaseModel((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = BaseModel((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        ness = NESS(self.model, (300, 300))

        ness = ness.longfilter(y[:500])

        x_pred, y_pred = ness.predict(50)

        for i in range(len(y_pred)):
            lower = np.percentile(y_pred[i], 1)
            upper = np.percentile(y_pred[i], 99)

            assert (y[500 + i] >= lower) and (y[500 + i] <= upper)

    def test_SMC2(self):
        x, y = self.model.sample(500)

        linear = BaseModel((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = BaseModel((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        smc2 = SMC2(self.model, (300, 300))

        smc2 = smc2.longfilter(y)

        weights = normalize(smc2._recw)

        values = smc2._filter._model.hidden.theta[1].values

        mean = np.average(values, weights=weights[:, None])
        std = np.sqrt(np.average((values - mean) ** 2, weights=weights[:, None]))

        assert mean - std < 1 < mean + std

    def test_NESSMC2(self):
        x, y = self.model.sample(500)

        linear = BaseModel((f0, g0), (f, g), (1, Gamma(1)), (Normal(), Normal()))

        self.model.hidden = linear
        self.model.observable = BaseModel((f0, g0), (fo, go), (1, Gamma(1)), (Normal(), Normal()))
        ness = NESSMC2(self.model, (1000, 100), filt=Linearized)

        ness = ness.longfilter(y)

        estimates = ness._filter._model.hidden.theta[1]

        mean = np.mean(estimates.values)
        std = np.std(estimates.values)

        assert mean - std < 1 < mean + std