import unittest
import numpy as np
import pykalman
from torch.distributions import Normal, Exponential, Independent
from pyfilter.filters import SISR, APF, UKF
from pyfilter.timeseries import AffineProcess, LinearGaussianObservations
from pyfilter.algorithms import NESS, SMC2, NESSMC2, IteratedFilteringV2
import torch
from pyfilter.proposals import Unscented


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return torch.tensor(0.)


def g0(alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x


def go(x, alpha, sigma):
    return sigma


def fmvn(x, alpha, sigma):
    x1 = alpha * x[0] + x[1] / 3
    x2 = x[1]
    return x1, x2


def gmvn(x, alpha, sigma):
    return sigma, sigma


def f0mvn(alpha, sigma):
    return torch.zeros(2)


def g0mvn(alpha, sigma):
    return sigma, sigma


class Tests(unittest.TestCase):
    # ===== Simple 1D model ===== #
    norm = Normal(0., 1.)
    linear = AffineProcess((f0, g0), (f, g), (1., 1.), (norm, norm))
    model = LinearGaussianObservations(linear, 1., 1.)

    # ===== Simple 2D model ===== #
    mvn = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
    mvn = AffineProcess((f0mvn, g0mvn), (fmvn, gmvn), (0.5, 1.), (mvn, mvn))
    a = torch.Tensor([1., 2.])

    mvnmodel = LinearGaussianObservations(mvn, a, 1.)

    def test_InitializeFilter(self):
        filt = SISR(self.model, 1000).initialize()

        assert filt._x_cur.shape == (1000,)

    def test_Filters(self):
        for model in [self.model, self.mvnmodel]:
            x, y = model.sample_path(500)

            for filter_, props in [
                (SISR, {'particles': 500}),
                (APF, {'particles': 500}),
                (UKF, {})
            ]:
                filt = filter_(model, **props).initialize()

                filt = filt.longfilter(y)

                assert len(filt.s_mx) > 0

                filtmeans = filt.filtermeans.numpy()

                # ===== Run Kalman ===== #
                if model is self.model:
                    kf = pykalman.KalmanFilter(transition_matrices=1., observation_matrices=1.)
                else:
                    kf = pykalman.KalmanFilter(transition_matrices=[[0.5, 1 / 3], [0, 1.]], observation_matrices=[1, 2])

                filterestimates = kf.filter(y.numpy())

                if filtmeans.ndim < 2:
                    filtmeans = filtmeans[:, None]

                rel_error = np.median(np.abs((filtmeans - filterestimates[0]) / filterestimates[0]))

                ll = kf.loglikelihood(y.numpy())

                rel_ll_error = np.abs((ll - np.array(filt.s_ll).sum()) / ll)

                assert rel_error < 0.05 and rel_ll_error < 0.05

    def test_ParallellFiltersAndStability(self):
        x, y = self.model.sample_path(50)

        shape = 30

        linear = AffineProcess((f0, g0), (f, g), (1., 1.), (self.norm, self.norm))
        self.model.hidden = linear

        filt = SISR(self.model, 1000).set_nparallel(shape).initialize().longfilter(y)

        filtermeans = filt.filtermeans

        x = filtermeans[:, :1]
        mape = ((x - filtermeans[:, 1:]) / x).abs()

        assert mape.median(0)[0].max() < 0.05

    def test_ParallelUnscented(self):
        x, y = self.model.sample_path(50)

        shape = 30

        linear = AffineProcess((f0, g0), (f, g), (1., 1.), (self.norm, self.norm))
        self.model.hidden = linear

        filt = SISR(self.model, 1000, proposal=Unscented()).set_nparallel(shape).initialize().longfilter(y)

        filtermeans = filt.filtermeans

        x = filtermeans[:, :1]
        mape = ((x - filtermeans[:, 1:]) / x).abs()

        assert mape.median(0)[0].max() < 0.05

    def test_Algorithms(self):
        priors = Exponential(2.), Exponential(2.)
        # ===== Test for multiple models ===== #
        hidden1d = AffineProcess((f0, g0), (f, g), priors, (self.linear.initial_dist, self.linear.increment_dist))
        oned = LinearGaussianObservations(hidden1d, 1., Exponential(1))

        hidden2d = AffineProcess((f0mvn, g0mvn), (fmvn, gmvn), priors, (self.mvn.initial_dist, self.mvn.increment_dist))
        twod = LinearGaussianObservations(hidden2d, self.a, Exponential(1.))

        # ====== Run inference ===== #
        for trumod, model in [(self.model, oned), (self.mvnmodel, twod)]:
            x, y = trumod.sample_path(550)

            algs = [
                (NESS, {'particles': 1000, 'filter_': SISR(model.copy(), 200)}),
                (SMC2, {'particles': 1000, 'filter_': SISR(model.copy(), 200)}),
                (NESSMC2, {'particles': 1000, 'filter_': SISR(model.copy(), 200)}),
                (IteratedFilteringV2, {'filter_': SISR(model.copy(), 1000)})
            ]

            for alg, props in algs:
                alg = alg(**props).initialize()

                alg = alg.fit(y)

                parameter = alg.filter.ssm.hidden.theta[-1]

                kde = parameter.get_kde(transformed=False)

                tru_val = trumod.hidden.theta[-1]
                densval = kde.logpdf(tru_val.numpy().reshape(-1, 1))
                priorval = parameter.distr.log_prob(tru_val)

                assert bool(densval > priorval.numpy())
