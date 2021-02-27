import unittest
from torch.distributions import Normal, Exponential, Independent, LogNormal
import torch
from scipy.stats import gaussian_kde
from pyfilter.filters import UKF, APF
from pyfilter.distributions import Prior
from pyfilter.timeseries import AffineProcess, LinearGaussianObservations
from pyfilter.utils import concater
from pyfilter.distributions import DistributionWrapper
from pyfilter.inference.sequential import NESSMC2, NESS, SMC2FW, SMC2
from pyfilter.inference.batch.variational import approximation as apx, VariationalBayes
from pyfilter.inference.batch.mcmc import PMMH


def f(x, alpha, sigma):
    return alpha * x.state


def g(x, alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x


def go(x, alpha, sigma):
    return sigma


def fmvn(x, alpha, sigma):
    x1 = alpha * x.state[..., 0]
    x2 = x.state[..., 1]
    return concater(x1, x2)


def gmvn(x, alpha, sigma):
    return concater(sigma, sigma)


def make_model(prob, dim=1):
    if prob:
        parameters = Prior(Exponential, rate=1.0), Prior(LogNormal, loc=0.0, scale=1.0)
    else:
        parameters = (0.99, 0.25) if dim == 1 else (0.5, 0.25)

    obs_param = dict()
    if dim == 1:
        dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)

        obs_param["a"] = 1.0
        obs_param["scale"] = 0.1

        func = (f, g)
    else:
        dist = DistributionWrapper(lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(2), scale=torch.ones(2))

        obs_param["a"] = torch.eye(2)
        obs_param["scale"] = 0.1 * torch.ones(2)

        func = (fmvn, gmvn)

    hidden = AffineProcess(func, parameters, dist, dist)

    return LinearGaussianObservations(hidden, **obs_param)


class InferenceAlgorithmTests(unittest.TestCase):
    def test_SequentialAlgorithms(self):
        particles = 1000

        for true_model, model in [(make_model(False), make_model(True)), (make_model(False, 2), make_model(True, 2))]:
            x, y = true_model.sample_path(1000)

            algs = [
                (NESS, {"particles": particles, "filter_": APF(model.copy(), 200)}),
                (NESS, {"particles": particles, "filter_": UKF(model.copy())}),
                (SMC2, {"particles": particles, "filter_": APF(model.copy(), 125)}),
                (SMC2FW, {"particles": particles, "filter_": APF(model.copy(), 200)}),
                (NESSMC2, {"particles": particles, "filter_": APF(model.copy(), 200)}),
            ]

            for alg_type, props in algs:
                alg = alg_type(**props)
                state = alg.fit(y)

                w = state.normalized_weights()

                zipped = zip(true_model.hidden.functional_parameters(), alg.filter.ssm.parameters_and_priors())

                for true_p, (p, prior) in zipped:
                    kde = gaussian_kde(prior.get_unconstrained(p).squeeze().numpy(), weights=w.numpy())

                    inverse_true_value = prior.bijection.inv(true_p)

                    posterior_log_prob = kde.logpdf(inverse_true_value.numpy().reshape(-1, 1))
                    prior_log_prob = prior.unconstrained_prior.log_prob(inverse_true_value)

                    assert (posterior_log_prob > prior_log_prob.numpy()).all()

    def test_VariationalBayes(self):
        static_model = make_model(False)
        x, y = static_model.sample_path(1000)

        model = make_model(True)

        vb = VariationalBayes(model, samples=12, max_iter=50_000)
        state = vb.fit(y, param_approx=apx.ParameterMeanField(), state_approx=apx.StateMeanField())

        assert state.converged

        # TODO: Check true values, not just convergence...

    def test_PMMH(self):
        static_model = make_model(False)
        x, y = static_model.sample_path(100)

        model = make_model(True)

        filt = APF(model, 200)
        pmmh = PMMH(filt, 500, num_chains=6)

        state = pmmh.fit(y)

        # TODO: Add check for posterior

    def test_PMMHCuda(self):
        if not torch.cuda.is_available():
            self.assertFalse(True)

        static_model = make_model(False)
        x, y = static_model.sample_path(100)

        model = make_model(True)

        filt = APF(model, 500)
        pmmh = PMMH(filt, 100, num_chains=6).to("cuda:0")

        state = pmmh.fit(y)


if __name__ == "__main__":
    unittest.main()
