import unittest
import numpy as np
import pykalman
from torch.distributions import Normal, Independent
from pyfilter.filters import SISR, APF, UKF
from pyfilter.timeseries import AffineProcess, LinearGaussianObservations, AffineEulerMaruyama
import torch
from pyfilter.proposals import Unscented, Linearized, Bootstrap
from pyfilter.utils import concater


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
    norm = Normal(0., 1.)
    linear = AffineProcess((f, g), (1., 1.), norm, norm)
    model = LinearGaussianObservations(linear, 1., 1.)

    # ===== Simple 2D model ===== #
    mvn = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
    mvn = AffineProcess((fmvn, gmvn), (0.5, 1.), mvn, mvn)
    a = torch.Tensor([1., 2.])

    mvnmodel = LinearGaussianObservations(mvn, a, 1.)

    def test_InitializeFilter(self):
        filt = SISR(self.model, 1000).initialize()

        assert filt._x_cur.shape == torch.Size([1000])

    def test_Filters(self):
        for model in [self.model, self.mvnmodel]:
            x, y = model.sample_path(500)

            for filter_, props in [
                (SISR, {'particles': 500}),
                (APF, {'particles': 500}),
                (UKF, {}),
                (SISR, {'particles': 500, 'proposal': Linearized(alpha=None)}),
                (APF, {'particles': 500, 'proposal': Linearized()}),
                (SISR, {'particles': 50, 'proposal': Unscented()})
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

        shape = 3000

        linear = AffineProcess((f, g), (1., 1.), self.norm, self.norm)
        self.model.hidden = linear

        filt = SISR(self.model, 1000).set_nparallel(shape).initialize().longfilter(y)

        filtermeans = filt.filtermeans

        x = filtermeans[:, :1]
        mape = ((x - filtermeans[:, 1:]) / x).abs()

        assert mape.median(0)[0].max() < 0.05

    def test_ParallelUnscented(self):
        x, y = self.model.sample_path(50)

        shape = 30

        linear = AffineProcess((f, g), (1., 1.), self.norm, self.norm)
        self.model.hidden = linear

        filt = SISR(self.model, 1000, proposal=Unscented()).set_nparallel(shape).initialize().longfilter(y)

        filtermeans = filt.filtermeans

        x = filtermeans[:, :1]
        mape = ((x - filtermeans[:, 1:]) / x).abs()

        assert mape.median(0)[0].max() < 0.05

    def test_SDE(self):
        def f(x, a, s):
            return -a * x

        def g(x, a, s):
            return s

        em = AffineEulerMaruyama((f, g), (0.02, 0.15), Normal(0., 1.), Normal(0., 1.), dt=1e-2, num_steps=10)
        model = LinearGaussianObservations(em, scale=1e-3)

        x, y = model.sample_path(500)

        with self.assertRaises(NotImplementedError):
            SISR(model, 200)

        for filt in [SISR(model, 500, proposal=Bootstrap()), UKF(model)]:
            filt = filt.initialize().longfilter(y)

            means = filt.filtermeans
            if isinstance(filt, UKF):
                means = means[:, 0]

            self.assertLess(torch.std(x - means), 5e-2)
