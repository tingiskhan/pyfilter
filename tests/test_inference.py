import pytest
from pyfilter.timeseries import LinearGaussianObservations, models as m, AffineObservations, StateSpaceModel
from pyfilter.distributions import Prior, DistributionWrapper
from torch.distributions import Normal, Exponential, LogNormal
from tests.test_filters import construct_filters
from pyfilter.inference.sequential import NESS, SMC2, SMC2FW, NESSMC2, threshold
from scipy.stats import gaussian_kde
from pyfilter.inference.batch import variational, mcmc
import torch
from pyfilter.filters import ParticleFilter
from pyfilter import collectors as colls


@pytest.fixture
def uhlenbecks():
    ou = m.OrnsteinUhlenbeck(0.025, 0.0, 0.05, dt=1.0)

    ou_priors = (
        Prior(Exponential, rate=1.0),
        Prior(Normal, loc=0.0, scale=1.0),
        Prior(LogNormal, loc=0.0, scale=1.0)
    )
    prob_ou = m.OrnsteinUhlenbeck(*ou_priors, dt=ou._dt.clone())

    return ou, prob_ou


@pytest.fixture
def models(uhlenbecks):
    ou, prob_ou = uhlenbecks

    a = 1.0
    s = 0.05

    obs_1d = LinearGaussianObservations(ou, a, s)
    prob_obs_1d = LinearGaussianObservations(prob_ou, a, s)

    return (
        [prob_obs_1d, obs_1d],
    )


@pytest.fixture
def exog_model(uhlenbecks):
    ou, ou_prob = uhlenbecks

    def _f(x, sigma):
        return x.exog + x.values

    def _g(x, sigma):
        return sigma

    normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
    obs = AffineObservations((_f, _g), (0.15,), normal)

    return StateSpaceModel(ou, obs), StateSpaceModel(ou_prob, obs)


def get_prior(name, ssm):
    if name.startswith("hidden"):
        module = ssm.hidden
    else:
        module = ssm.observable

    return module.prior_dict[name.split(".")[-1]]


def get_true_parameter(name, model):
    if name.startswith("hidden"):
        module = model.hidden
    else:
        module = model.observable

    return module.parameters_and_buffers()[name.split(".")[-1]]


def check_posterior(model, true_model, **kde_kwargs):
    for name, parameter in model.named_parameters():
        prior = get_prior(name, model)
        true_parameter = get_true_parameter(name, true_model)

        unconstrained = prior.get_unconstrained(parameter)
        if parameter.dim() == 2:
            unconstrained.squeeze_(1)

        kde = gaussian_kde(unconstrained.numpy(), **kde_kwargs)

        inverse_true_value = prior.bijection.inv(true_parameter)

        posterior_log_prob = kde.logpdf(inverse_true_value)
        prior_log_prob = prior.unconstrained_prior.log_prob(inverse_true_value).numpy()

        assert posterior_log_prob > prior_log_prob


class TestThresholds(object):
    def test_constant_threshold(self):
        thresh = 0.5
        t = threshold.ConstantThreshold(thresh)

        for i in range(500):
            assert t.get_threshold(i) == thresh

    def test_decaying_threshold(self):
        start_thresh = 0.5
        min_thresh = 0.1

        half_life = 50
        t = threshold.DecayingThreshold(min_thresh, start_thresh, half_life)

        for i in range(100):
            if i == half_life:
                assert t.get_threshold(i) == (start_thresh / 2.0)

    def test_interval_threshold(self):
        min_thresh = 0.1

        thresholds = {10: 0.5, 50: 0.2, 100: 0.15}
        t = threshold.IntervalThreshold(thresholds, min_thresh)

        for i in range(105):
            thresh = t.get_threshold(i)

            if i <= 10:
                assert thresh == thresholds[10]
            elif i <= 50:
                assert thresh == thresholds[50]
            elif i <= 100:
                assert thresh == thresholds[100]
            else:
                assert thresh == min_thresh


class TestsSequentialAlgorithm(object):
    PARTICLES = 2_000
    SERIES_LENGTH = 1_000

    def sequential_algorithms(self, filter_, **kwargs):
        yield NESS(filter_, **kwargs)
        yield SMC2(filter_, **kwargs)

        decay_thresh = threshold.DecayingThreshold(half_life=self.SERIES_LENGTH // 2, start_thresh=0.5, min_thresh=0.2)
        yield SMC2(filter_, **kwargs, threshold=decay_thresh)

        thresholds = {100: 0.5, 500: 0.25}
        interval_thresh = threshold.IntervalThreshold(thresholds, ending_threshold=0.2)
        yield SMC2(filter_, **kwargs, threshold=interval_thresh)

        yield SMC2FW(filter_, **kwargs)
        yield NESSMC2(filter_, **kwargs)

    def test_algorithms(self, models):
        for prob_model, model in models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(prob_model, particles=250):
                for algorithm in self.sequential_algorithms(f.copy(), particles=self.PARTICLES):
                    result = algorithm.fit(y)

                    check_posterior(algorithm.filter.ssm, model, weights=result.normalized_weights().numpy())

    def test_exog(self, exog_model):
        model, prob_model = exog_model

        model.observable.exog = prob_model.observable.exog = torch.arange(0, self.SERIES_LENGTH + 1)
        x, y = model.sample_path(self.SERIES_LENGTH)

        assert (y.diff() > 0.0).all()

        for f in construct_filters(prob_model):
            for algorithm in self.sequential_algorithms(f.copy(), particles=self.PARTICLES):
                result = algorithm.fit(y)

                check_posterior(algorithm.filter.ssm, model, weights=result.normalized_weights().numpy())

    def test_collectors(self, models):
        for prob_model, model in models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(prob_model, particles=250):
                for algorithm in self.sequential_algorithms(f.copy(), particles=self.PARTICLES):
                    algorithm.register_forward_hook(colls.MeanCollector())
                    algorithm.register_forward_hook(colls.ParameterPosterior())

                    is_particle_filter = isinstance(f, ParticleFilter)
                    if is_particle_filter:
                        algorithm.register_forward_hook(colls.Standardizer())

                    result = algorithm.fit(y)

                    assert "filter_means" in result.tensor_tuples
                    if is_particle_filter:
                        assert "standardized" in result.tensor_tuples


class TestBatchAlgorithms(object):
    SERIES_LENGTH = 500
    BURN_IN = 500
    MONTE_CARLO_SAMPLES = 5_000

    def test_variational_bayes(self, models):
        for prob_model, model in models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            algorithm = variational.VariationalBayes(
                prob_model,
                n_samples=12,
                max_iter=100_000,
                parameter_approximation=variational.approximation.MeanField(),
                state_approximation=variational.approximation.MeanField()
            )

            result: variational.VariationalResult = algorithm.fit(y)

            assert result.converged

            result.sample_and_update_parameters(algorithm.model, torch.Size([self.MONTE_CARLO_SAMPLES]), ignore_grad=True)
            check_posterior(algorithm.model, model)

    @staticmethod
    def pmmh_proposals(filter_, **kwargs):
        yield mcmc.PMMH(filter_, proposal=mcmc.proposals.RandomWalk(scale=0.05), initializer="mean", **kwargs)
        yield mcmc.PMMH(filter_, proposal=mcmc.proposals.RandomWalk(scale=0.05), **kwargs)

        if isinstance(filter_, ParticleFilter):
            yield mcmc.PMMH(filter_, proposal=mcmc.proposals.GradientBasedProposal(scale=0.05), **kwargs)
            yield mcmc.PMMH(filter_, proposal=mcmc.proposals.GradientBasedProposal(scale=0.025), **kwargs)

    def test_pmmh(self, models):
        for prob_model, model in models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(prob_model):
                for algorithm in self.pmmh_proposals(f.copy(), samples=2 * self.BURN_IN, num_chains=4):
                    result: mcmc.PMMHResult = algorithm.fit(y)

                    result.update_parameters_from_chain(algorithm.filter.ssm, self.BURN_IN)
                    check_posterior(algorithm.filter.ssm, model)

    def test_mean_field_approximation(self):
        for size in [torch.Size([10]), torch.Size([1000, 5])]:
            mean_field = variational.approximation.MeanField()

            mean_field.initialize(size)

            assert (
                    (mean_field.mean.shape == size) and
                    (mean_field.log_std.shape == size) and
                    (mean_field._independent_dim == len(size))
            )