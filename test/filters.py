import unittest
import numpy as np
import pykalman
from math import sqrt
from torch.distributions import Normal, Independent
from pyfilter.filters import SISR, APF, UKF, proposals as prop
from pyfilter.timeseries import AffineProcess, LinearGaussianObservations, AffineEulerMaruyama
import torch
from pyfilter.utils import concater
from pyfilter.distributions import DistributionWrapper


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x


def go(x, alpha, sigma):
    return sigma


def fmvn(x, alpha, sigma):
    x1 = alpha * x[..., 0] + x[..., 1] / 3
    x2 = x[..., 1]
    return concater(x1, x2)


def gmvn(x, alpha, sigma):
    return concater(sigma, sigma)


class Tests(unittest.TestCase):
    # ===== Simple 1D model ===== #
    norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
    linear = AffineProcess((f, g), (1., 1.), norm, norm)
    model = LinearGaussianObservations(linear, 1., 1.)

    # ===== Simple 2D model ===== #
    mvn = DistributionWrapper(lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(2), scale=torch.ones(2))
    mvn = AffineProcess((fmvn, gmvn), (0.5, 1.), mvn, mvn)
    a = torch.tensor([1., 2.])

    mvnmodel = LinearGaussianObservations(mvn, a, 1.)

    def test_InitializeFilter(self):
        state = SISR(self.model, 1000).initialize()

        assert state.x.shape == torch.Size([1000])

    def test_Filters(self):
        for model in [self.model, self.mvnmodel]:
            x, y = model.sample_path(500)

            for filter_type, props in [
                (SISR, {'particles': 500}),
                (APF, {'particles': 500}),
                (UKF, {}),
                (SISR, {'particles': 50, 'proposal': prop.Unscented()})
            ]:
                filt = filter_type(model, **props)
                result = filt.longfilter(y, record_states=True)

                filtmeans = result.filter_means.numpy()

                # ===== Run Kalman ===== #
                if model is self.model:
                    kf = pykalman.KalmanFilter(transition_matrices=1., observation_matrices=1.)
                else:
                    kf = pykalman.KalmanFilter(transition_matrices=[[0.5, 1 / 3], [0, 1.]], observation_matrices=[1, 2])

                f_mean, _ = kf.filter(y.numpy())

                if model.hidden_ndim < 1 and not isinstance(filt, UKF):
                    f_mean = f_mean[:, 0]

                rel_error = np.median(np.abs((filtmeans - f_mean) / f_mean))

                ll = kf.loglikelihood(y.numpy())
                rel_ll_error = np.abs((ll - result.loglikelihood.numpy()) / ll)

                assert rel_error < 0.05 and rel_ll_error < 0.05

    def test_ParallellFiltersAndStability(self):
        x, y = self.model.sample_path(50)

        shape = 3000

        linear = AffineProcess((f, g), (1., 1.), self.norm, self.norm)
        self.model.hidden = linear

        filt = SISR(self.model, 1000).set_nparallel(shape)
        result = filt.longfilter(y)

        filtermeans = result.filter_means

        x = filtermeans[:, :1]
        mape = ((x - filtermeans[:, 1:]) / x).abs()

        assert mape.median(0)[0].max() < 0.05

    def test_ParallelUnscented(self):
        x, y = self.model.sample_path(50)

        shape = 30

        linear = AffineProcess((f, g), (1., 1.), self.norm, self.norm)
        self.model.hidden = linear

        filt = SISR(self.model, 1000, proposal=prop.Unscented()).set_nparallel(shape)
        result = filt.longfilter(y)

        filtermeans = result.filter_means

        x = filtermeans[:, :1]
        mape = ((x - filtermeans[:, 1:]) / x).abs()

        assert mape.median(0)[0].max() < 0.05

    def test_SDE(self):
        def f(x, a, s):
            return -a * x

        def g(x, a, s):
            return s

        dt = 1e-2
        norm = DistributionWrapper(Normal, loc=0.0, scale=sqrt(dt))

        em = AffineEulerMaruyama((f, g), (0.02, 0.15), norm, norm, dt=1e-2, num_steps=10)
        model = LinearGaussianObservations(em, scale=1e-3)

        x, y = model.sample_path(500)

        for filt in [SISR(model, 500, proposal=prop.Bootstrap()), UKF(model)]:
            result = filt.longfilter(y)

            means = result.filter_means
            if isinstance(filt, UKF):
                means = means[:, 0]

            self.assertLess(torch.std(x - means), 5e-2)
