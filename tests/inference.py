import pytest
from pyfilter.timeseries import LinearGaussianObservations, models as m
from pyfilter.distributions import Prior
from torch.distributions import Normal, Exponential, LogNormal
from tests.filters import construct_filters
from pyfilter.inference.sequential import NESS, SMC2, SMC2FW, NESSMC2
from scipy.stats import gaussian_kde
from pyfilter.inference.batch.variational import VariationalBayes, approximation as apx


# @pytest.fixture
def models():
    ou = m.OrnsteinUhlenbeck(0.01, 0.0, 0.05, dt=1.0)
    obs_1d = LinearGaussianObservations(ou, 1.0, 0.05)

    ou_priors = (
        Prior(Exponential, rate=5.0),
        Prior(Normal, loc=0.0, scale=1.0),
        Prior(LogNormal, loc=0.0, scale=1.0)
    )
    prob_ou = m.OrnsteinUhlenbeck(*ou_priors, dt=ou._dt.clone())
    prob_obs_1d = LinearGaussianObservations(prob_ou, obs_1d.observable.parameter_0, obs_1d.observable.parameter_1)

    return (
        [prob_obs_1d, obs_1d],
    )


def sequential_algorithms(filter_, **kwargs):
    return (
        NESS(filter_, **kwargs),
        SMC2(filter_, **kwargs),
        SMC2FW(filter_, **kwargs),
        NESSMC2(filter_, **kwargs),
    )


def get_prior(name, algorithm):
    return next(o for n, o in algorithm.named_modules() if name.split(".")[-1] in n)


def get_true_parameter(name, model):
    return next(o for n, o in model.named_buffers() if n == name)


class TestsSequentialAlgorithm(object):
    PARTICLES = 1000
    SERIES_LENGTH = 1000

    def test_sequential_algorithms(self, models):
        for prob_model, model in models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(prob_model):
                for algorithm in sequential_algorithms(f.copy(), particles=self.PARTICLES):
                    result = algorithm.fit(y)

                    w = result.normalized_weights()

                    for name, parameter in algorithm.filter.ssm.named_parameters():
                        prior = get_prior(name, algorithm)
                        true_parameter = get_true_parameter(name, model)

                        kde = gaussian_kde(prior.get_unconstrained(parameter).squeeze().numpy(), weights=w.numpy())

                        inverse_true_value = prior.bijection.inv(true_parameter)

                        posterior_log_prob = kde.logpdf(inverse_true_value)
                        prior_log_prob = prior.unconstrained_prior.log_prob(inverse_true_value).numpy()

                        assert (posterior_log_prob > prior_log_prob).all()


class TestBatchAlgorithms(object):
    def test_variational_bayes(self, models):
        for prob_model, model in models:
            x, y = model.sample_path(TestsSequentialAlgorithm.SERIES_LENGTH)

            algorithm = VariationalBayes(
                prob_model,
                n_samples=12,
                max_iter=100_000,
                parameter_approximation=apx.ParameterMeanField(),
                state_approximation=apx.StateMeanField()
            )

            result = algorithm.fit(y)

            assert result.converged

            for name, parameter in algorithm._model.named_parameters():
                prior = get_prior(name, algorithm)
                true_parameter = get_true_parameter(name, model)

            # TODO: Add check for distribution

TestBatchAlgorithms().test_variational_bayes(models())
# TODO: Add test for PMMH with all available proposals
