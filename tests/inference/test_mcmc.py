import pytest
from pyfilter import inference as inf, filters as filts
from pyro.distributions import Normal, Beta, LogNormal
from stochproc import timeseries as ts
from .models import linear_models, build_obs_1d


class TestPMCMC(object):
    @pytest.mark.parametrize("true_model", linear_models())
    def test_pmcmc(self, true_model):
        _, y = true_model.sample_states(100).get_paths()

        with inf.make_context() as cntxt:
            alpha = cntxt.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = cntxt.named_parameter("beta", inf.Prior(Beta, concentration1=5.0, concentration0=1.0))
            sigma = cntxt.named_parameter("sigma", inf.Prior(LogNormal, loc=-2.0, scale=0.5))

            prob_model = ts.models.AR(alpha, beta, sigma)
            ssm = build_obs_1d(prob_model)

            filter_ = filts.APF(ssm, 500)
            pmcmc = inf.batch.mcmc.PMMH(filter_, 500, initializer="mean")

            result = pmcmc.fit(y)

            print()

