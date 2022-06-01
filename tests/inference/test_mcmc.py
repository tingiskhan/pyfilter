import itertools

import pytest
from pyfilter import inference as inf, filters as filts
from pyro.distributions import LogNormal
from stochproc import timeseries as ts
from .models import linear_models, build_obs_1d


def build_model(cntxt):
    sigma = cntxt.named_parameter("sigma", inf.Prior(LogNormal, loc=-2.0, scale=0.5))

    prob_model = ts.models.RandomWalk(scale=sigma)
    return build_obs_1d(prob_model)


def proposals():
    yield inf.batch.mcmc.proposals.GradientBasedProposal(scale=5e-2), True
    yield inf.batch.mcmc.proposals.RandomWalk(scale=5e-2), False


PARAMETERS = itertools.product(linear_models(), proposals())


class TestPMCMC(object):
    @pytest.mark.parametrize("true_model, kernel_and_record_states", PARAMETERS)
    def test_pmcmc(self, true_model, kernel_and_record_states):
        _, y = true_model.sample_states(500).get_paths()

        with inf.make_context():
            kernel, record_states = kernel_and_record_states
            filter_ = filts.APF(build_model, 150, record_states=record_states)
            pmcmc = inf.batch.mcmc.PMMH(filter_, 1_000, initializer="mean", proposal=kernel)

            result = pmcmc.fit(y)

            # TODO: Add something to test
            print()