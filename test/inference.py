import unittest
from pyfilter.inference import algorithms as a
from torch.distributions import Normal, Exponential, Independent, LogNormal
from pyfilter.filters import UKF, APF
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
    x1 = alpha * x[..., 0]
    x2 = x[..., 1]
    return concater(x1, x2)


def gmvn(x, alpha, sigma):
    return concater(sigma, sigma)


class MyTestCase(unittest.TestCase):
    def test_Inference(self):
        # ===== Distributions ===== #
        dist = Normal(0., 1.)
        mvn = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

        # ===== Define model ===== #
        linear = AffineProcess((f, g), (1., 0.25), dist, dist)
        model = LinearGaussianObservations(linear, scale=0.1)

        mv_linear = AffineProcess((fmvn, gmvn), (0.5, 0.25), mvn, mvn)
        mvnmodel = LinearGaussianObservations(mv_linear, torch.eye(2), scale=0.1)

        # ===== Test for multiple models ===== #
        priors = Exponential(1.), LogNormal(0., 1.)

        hidden1d = AffineProcess((f, g), priors, dist, dist)
        oned = LinearGaussianObservations(hidden1d, 1., scale=0.1)

        hidden2d = AffineProcess((fmvn, gmvn), priors, mvn, mvn)
        twod = LinearGaussianObservations(hidden2d, torch.eye(2), scale=0.1 * torch.ones(2))

        particles = 1000
        # ====== Run inference ===== #
        for trumod, model in [(model, oned), (mvnmodel, twod)]:
            x, y = trumod.sample_path(1000)

            algs = [
                (a.sequential.NESS, {'particles': particles, 'filter_': APF(model.copy(), 200)}),
                (a.sequential.NESS, {'particles': particles, 'filter_': UKF(model.copy())}),
                (a.sequential.SMC2, {'particles': particles, 'filter_': APF(model.copy(), 200)}),
                (a.sequential.SMC2FW, {'particles': particles, 'filter_': APF(model.copy(), 200)}),
                (a.sequential.NESSMC2, {'particles': particles, 'filter_': APF(model.copy(), 200)})
            ]

            for alg, props in algs:
                alg = alg(**props)
                alg.fit(y)

                w = normalize(alg._w_rec if hasattr(alg, '_w_rec') else torch.ones(particles))

                tru_params = trumod.hidden.theta._cont + trumod.observable.theta._cont
                inf_params = alg.filter.ssm.hidden.theta._cont + alg.filter.ssm.observable.theta._cont

                for trup, p in zip(tru_params, inf_params):
                    if not p.trainable:
                        continue

                    kde = p.get_kde(weights=w)

                    transed = p.bijection.inv(trup)
                    densval = kde.logpdf(transed.numpy().reshape(-1, 1))
                    priorval = p.distr.log_prob(trup)

                    assert (densval > priorval.numpy()).all()


if __name__ == '__main__':
    unittest.main()
