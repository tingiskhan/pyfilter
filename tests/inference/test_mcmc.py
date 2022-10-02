import pytest
from pyfilter import inference as inf, filters as filts
from .models import linear_models


def proposals():
    yield inf.batch.mcmc.proposals.GradientBasedProposal(scale=5e-2), True
    yield inf.batch.mcmc.proposals.RandomWalk(scale=5e-2), False


class TestPMCMC(object):
    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("kernel_and_record_states", proposals())
    def test_pmcmc(self, models, kernel_and_record_states):
        true_model, build_model = models
        _, y = true_model.sample_states(50).get_paths()

        with inf.make_context() as context:
            kernel, record_states = kernel_and_record_states
            filter_ = filts.APF(lambda u: build_model(u, use_cuda=False), 150, record_states=record_states)

            # TODO: Just make sure it runs
            pmcmc = inf.batch.mcmc.PMMH(filter_, 10, initializer="mean", proposal=kernel)

            result = pmcmc.fit(y)

            # TODO: Add something to test
            print()