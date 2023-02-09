import pytest
import torch

from pyfilter.filters import particle as part
import pyro
from pyro.infer import autoguide, SVI
from pyro.optim import Adam
from pyro.distributions import LogNormal
from stochproc import timeseries as ts

from pyfilter.filters.particle import proposals


def do_infer_with_pyro(model, data, num_samples=1_000, niter=250, num_particles=1):
    guide = autoguide.AutoDiagonalNormal(model)
    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True))

    pyro.clear_param_store()

    for n in range(niter):
        loss = svi.step(data)

    posterior_predictive = pyro.infer.Predictive(
        model,
        guide=guide,
        num_samples=num_samples
    )

    return guide, posterior_predictive


class TestPyroIntegration(object):
    @pytest.mark.parametrize("num_particles", [1, 4])
    def test_vi(self, num_particles):
        def build_ssm(sigma_):
            latent = ts.models.RandomWalk(sigma_)
            return ts.LinearStateSpaceModel(latent, (1.0, 0.0, 0.1), torch.Size([]))

        def pyro_model(y, sample_filter=True):
            sigma = pyro.sample("sigma", LogNormal(0.0, 1.0))
            ssm = build_ssm(sigma)

            if not sample_filter:
                return
            
            filt = part.APF(ssm, 100, record_states=True, proposal=proposals.LinearGaussianObservations())
            filt.set_batch_shape(sigma.shape)
            filt.do_sample_pyro(y, pyro)

        torch.manual_seed(123)
        true_sigma = 0.05
        ssm = build_ssm(true_sigma)

        _, y = ssm.sample_states(250).get_paths()

        _, posterior_predictive = do_infer_with_pyro(pyro_model, y, niter=500, num_particles=num_particles)
        posterior_draws = posterior_predictive(y, sample_filter=False)

        mean = posterior_draws["sigma"].mean()
        std = posterior_draws["sigma"].std()

        assert (mean - 2.0 * std) <= true_sigma <= (mean + 2.0 * std)
