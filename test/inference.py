import unittest
from torch.distributions import Normal, Exponential, Independent, LogNormal
from pyfilter.filters import UKF, APF
from pyfilter.timeseries import AffineProcess, LinearGaussianObservations
from pyfilter.utils import concater
from pyfilter.normalization import normalize
import torch
from pyfilter.inference.sequential import NESSMC2, NESS, SMC2FW, SMC2
from pyfilter.inference.batch.variational import approximation as apx, VariationalBayes
from pyfilter.inference.batch.mcmc import PMMH
from scipy.stats import gaussian_kde


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


class InferenceAlgorithmTests(unittest.TestCase):
    def test_SequentialAlgorithms(self):
        # ===== Distributions ===== #
        dist = Normal(0., 1.)
        mvn = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

        # ===== Define model ===== #
        linear = AffineProcess((f, g), (0.99, 0.25), dist, dist)
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
                (NESS, {'particles': particles, 'filter_': APF(model.copy(), 200)}),
                (NESS, {'particles': particles, 'filter_': UKF(model.copy())}),
                (SMC2, {'particles': particles, 'filter_': APF(model.copy(), 200)}),
                (SMC2FW, {'particles': particles, 'filter_': APF(model.copy(), 200)}),
                (NESSMC2, {'particles': particles, 'filter_': APF(model.copy(), 200)})
            ]

            for alg, props in algs:
                alg = alg(**props)
                state = alg.fit(y)

                w = normalize(state.w)

                zipped = zip(
                    trumod.hidden.parameters + trumod.observable.parameters,                  # True parameter values
                    alg.filter.ssm.hidden.parameters + alg.filter.ssm.observable.parameters   # Inferred
                )

                for trup, p in zipped:
                    if not p.trainable:
                        continue

                    kde = gaussian_kde(p.t_values.numpy(), weights=w.numpy())

                    inverse_true_value = p.bijection.inv(trup)

                    posterior_log_prob = kde.logpdf(inverse_true_value.numpy().reshape(-1, 1))
                    prior_log_prob = p.bijected_prior.log_prob(inverse_true_value)

                    assert (posterior_log_prob > prior_log_prob.numpy()).all()

    def test_VariationalBayes(self):
        # ===== Distributions ===== #
        dist = Normal(0., 1.)

        # ===== Define model ===== #
        linear = AffineProcess((f, g), (0.99, 0.25), dist, dist)
        model = LinearGaussianObservations(linear, scale=0.1)

        # ===== Sample ===== #
        x, y = model.sample_path(1000)

        # ==== Construct model to train ===== #
        priors = Exponential(1.), LogNormal(0., 1.)

        hidden1d = AffineProcess((f, g), priors, dist, dist)
        oned = LinearGaussianObservations(hidden1d, 1., scale=0.1)

        vb = VariationalBayes(oned, samples=12, max_iter=50_000)
        state = vb.fit(y, param_approx=apx.ParameterMeanField(), state_approx=apx.StateMeanField())

        assert state.converged

        # TODO: Check true values, not just convergence...

    def test_PMMH(self):
        # ===== Distributions ===== #
        dist = Normal(0., 1.)

        # ===== Define model ===== #
        linear = AffineProcess((f, g), (0.99, 0.25), dist, dist)
        model = LinearGaussianObservations(linear, scale=0.1)

        # ===== Sample ===== #
        x, y = model.sample_path(100)

        # ==== Construct model to train ===== #
        priors = Exponential(1.), LogNormal(0., 1.)

        hidden1d = AffineProcess((f, g), priors, dist, dist)
        oned = LinearGaussianObservations(hidden1d, 1., scale=0.1)

        filt = APF(oned, 200)
        pmmh = PMMH(filt, 50, num_chains=6)

        state = pmmh.fit(y)
        oned.copy().parameters_from_array(state.as_tensor())
        print()
        # TODO: Add check for posterior


if __name__ == '__main__':
    unittest.main()
