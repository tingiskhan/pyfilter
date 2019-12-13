import unittest
from pyfilter.algorithms import NESS, SMC2, NESSMC2, IteratedFilteringV2
from torch.distributions import Normal, Exponential, Independent
from pyfilter.filters import SISR, UKF
from pyfilter.timeseries import AffineProcess, LinearGaussianObservations
from pyfilter.utils import concater
from pyfilter.normalization import normalize
import torch


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x


def go(x, alpha, sigma):
    return sigma


def fmvn(x, alpha, sigma):
    x1 = alpha * x[0] + x[1] / 3
    x2 = x[1]
    return concater(x1, x2)


def gmvn(x, alpha, sigma):
    return concater(sigma, sigma)


class MyTestCase(unittest.TestCase):
    def test_Algorithms(self):
        # ===== Distributions ===== #
        dist = Normal(0., 1.)
        mvn = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

        # ===== Define model ===== #
        linear = AffineProcess((f, g), (1., 0.25), dist, dist)
        model = LinearGaussianObservations(linear, scale=0.1)

        mv_linear = AffineProcess((fmvn, gmvn), (0.5, 0.25), mvn, mvn)
        mvnmodel = LinearGaussianObservations(mv_linear, scale=0.1)

        # ===== Test for multiple models ===== #
        priors = Normal(0., 1.), Exponential(1.)

        hidden1d = AffineProcess((f, g), priors, dist, dist)
        oned = LinearGaussianObservations(hidden1d, Normal(0., 1.), Exponential(1.))

        hidden2d = AffineProcess((fmvn, gmvn), priors, mvn, mvn)
        prior = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
        twod = LinearGaussianObservations(hidden2d, prior, Exponential(1.))

        # ====== Run inference ===== #
        for trumod, model in [(model, oned), (mvnmodel, twod)]:
            x, y = trumod.sample_path(1000)

            algs = [
                (NESS, {'particles': 1000, 'filter_': SISR(model.copy(), 200, ess=1.)}),
                (NESS, {'particles': 1000, 'filter_': UKF(model.copy())}),
                (SMC2, {'particles': 1000, 'filter_': SISR(model.copy(), 200, ess=1.)}),
                (NESSMC2, {'particles': 1000, 'filter_': SISR(model.copy(), 200, ess=1.)}),
                (IteratedFilteringV2, {'filter_': SISR(model.copy(), 1000)})
            ]

            for alg, props in algs:
                alg = alg(**props).initialize()

                alg = alg.fit(y)

                w = normalize(alg._w_rec)
                for trup, p in zip(trumod.hidden.theta + trumod.observable.theta, alg.filter.ssm.theta_dists):
                    kde = p.get_kde(weights=w)

                    transed = p.bijection.inv(trup)
                    densval = kde.logpdf(transed.numpy().reshape(-1, 1))
                    priorval = p.distr.log_prob(trup)

                    assert (densval > priorval.numpy()).all()


if __name__ == '__main__':
    unittest.main()
