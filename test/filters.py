import unittest
import numpy as np
import pykalman
from torch.distributions import Normal, Exponential, Independent
from pyfilter.filters import SISR, APF, UKF
from pyfilter.timeseries import StateSpaceModel, Observable, BaseModel
from pyfilter.algorithms import NESS, SMC2, NESSMC2
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
    mvn = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
    mvn = BaseModel((f0mvn, g0mvn), (fmvn, gmvn), (0.5, 1.), (mvn, mvn))
    mvnobs = Observable((fomvn, go), (1., 1.), norm)
    mvnmodel = StateSpaceModel(mvn, mvnobs)

    def test_InitializeFilter(self):
        filt = SISR(self.model, 1000).initialize()

        assert filt._x_cur.shape == (1000,)

    def test_Filters(self):
        for model in [self.model, self.mvnmodel]:
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

    def test_ParallellFilters(self):
        x, y = self.model.sample(50)

        shape = 1000, 1

        linear = BaseModel((f0, g0), (f, g), (torch.ones(shape), torch.ones(shape)), (self.norm, self.norm))
        self.model.hidden = linear

        filt = APF(self.model, (shape[0], 1000)).initialize().longfilter(y)

        filtermeans = torch.cat(filt.filtermeans()).reshape(x.shape[0], -1).numpy()

        rmse = np.sqrt(np.mean((filtermeans[:, 0:1] - filtermeans[:, 1:]) ** 2))

        assert rmse < 0.1

    def test_Algorithms(self):
        x, y = self.model.sample(500)

        hidden = BaseModel((f0, g0), (f, g), (1., Exponential(1)), (self.norm, self.norm))
        observable = Observable((fo, go), (1., Exponential(1)), self.norm)

        model = StateSpaceModel(hidden, observable)

        algs = [
            (NESS, {'particles': 1000, 'filter_': SISR(model, 400)}),
            (SMC2, {'particles': 1000, 'filter_': SISR(model, 400)}),
            (NESSMC2, {'particles': 1000, 'filter_': SISR(model, 400)})
        ]

        for alg, props in algs:
            alg = alg(**props).initialize()

            alg = alg.fit(y.numpy())

            parameter = alg.filter.ssm.hidden.theta[1]

            kde = parameter.get_kde()

            tru_val = self.model.hidden.theta_vals[-1]
            densval = kde.score_samples(tru_val)
            priorval = parameter.dist.log_prob(tru_val)

            assert bool(densval > priorval)
