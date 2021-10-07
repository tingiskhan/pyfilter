import pytest
from pyfilter.timeseries import LinearGaussianObservations, models as m
from pyfilter.distributions import Prior
from torch.distributions import Normal, Exponential, LogNormal
from tests.filters import construct_filters, TestFilters
from pyfilter.inference.sequential import NESS, SMC2, SMC2FW, NESSMC2


@pytest.fixture
def models():
    ou = m.OrnsteinUhlenbeck(0.01, 0.0, 0.05, dt=1.0)
    obs_1d = LinearGaussianObservations(ou, 1.0, 0.05)

    ou_priors = (
        Prior(Exponential, rate=1.0),
        Prior(Normal, loc=0.0, scale=1.0),
        Prior(LogNormal, loc=0.0, scale=1.0)
    )
    prob_ou = m.OrnsteinUhlenbeck(*ou_priors, dt=1.0)
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
    SERIES_LENGTH = 500

    def test_sequential_algorithms(self, models):
        for prob_model, model in models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(prob_model.copy()):
                for algorithm in sequential_algorithms(f, particles=self.PARTICLES):
                    result = algorithm.fit(y)

                    # TODO: Construct KDEs for parameters and compare with prior and assert that higher likelihood