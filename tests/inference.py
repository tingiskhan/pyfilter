import pytest
from pyfilter.timeseries import LinearGaussianObservations, models as m
from pyfilter.distributions import Prior
from torch.distributions import Normal, Exponential, LogNormal
from tests.filters import construct_filters
from pyfilter.inference.sequential import NESS, SMC2, SMC2FW, NESSMC2
from pyfilter.inference.utils import parameters_and_priors_from_model
from scipy.stats import gaussian_kde


@pytest.fixture
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

                    zipped = zip(
                        parameters_and_priors_from_model(algorithm.filter.ssm), model.hidden.functional_parameters()
                    )

                    for (parameter, prior), true_parameter in zipped:
                        kde = gaussian_kde(prior.get_unconstrained(parameter).squeeze().numpy(), weights=w.numpy())

                        inverse_true_value = prior.bijection.inv(true_parameter)

                        posterior_log_prob = kde.logpdf(inverse_true_value)
                        prior_log_prob = prior.unconstrained_prior.log_prob(inverse_true_value).numpy()

                        assert (posterior_log_prob > prior_log_prob).all()
