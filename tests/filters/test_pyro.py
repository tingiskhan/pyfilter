import pytest
import torch

from pyfilter.filters import particle as part
import pyro
from pyro.infer import autoguide, SVI
from pyro.optim import Adam
from pyro.distributions import LogNormal, Normal
from stochproc import timeseries as ts


def do_infer_with_pyro(model, data, num_samples=1_000, niter=250, num_particles=1):
    guide = autoguide.AutoDiagonalNormal(model)
    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO(num_particles=num_particles))

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
            return ts.StateSpaceModel(latent, lambda u: Normal(u.values, 0.1), ())

        def pyro_model(y):
            # TODO: I think the filter should keep track of when to unsqueeze etc...
            sigma = pyro.sample("sigma", LogNormal(0.0, 1.0))
            ssm = build_ssm(sigma)

            filt = part.APF(ssm, 100, record_states=True)
            filt.do_sample_pyro(y, pyro)

        true_sigma = 0.05
        ssm = build_ssm(true_sigma)

        x, y = ssm.sample_states(250).get_paths()

        guid, posterior_predictive = do_infer_with_pyro(pyro_model, x, niter=500, num_particles=num_particles)
        posterior_draws = posterior_predictive(x)

        mean = posterior_draws["sigma"].mean()
        std = posterior_draws["sigma"].std()
        print(mean, std)

        assert (mean - 2.0 * std) <= true_sigma <= (mean + 2.0 * std)
