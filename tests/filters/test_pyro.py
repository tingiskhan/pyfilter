from pyfilter.filters import particle as part
import pyro
from pyro.distributions import LogNormal, Normal
from stochproc import timeseries as ts


def do_infer_with_pyro(model, data, num_samples=1_000, niter=250):
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optim = pyro.optim.Adam({"lr": 0.01})
    svi = pyro.infer.SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO())

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
    def test_vi(self):
        def build_ssm(sigma_):
            latent = ts.models.RandomWalk(sigma_)
            return ts.StateSpaceModel(latent, lambda u: Normal(u.values, 0.1), ())

        def pyro_model(y):
            sigma = pyro.sample("sigma", LogNormal(0.0, 1.0))
            ssm = build_ssm(sigma)
            filt = part.APF(ssm, 100, record_states=True)

            filt.do_sample_pyro(y, pyro)

        true_sigma = 0.05
        ssm = build_ssm(true_sigma)

        x, y = ssm.sample_states(250).get_paths()

        guid, posterior_predictive = do_infer_with_pyro(pyro_model, x, niter=500)
        posterior_draws = posterior_predictive(x)

        mean = posterior_draws["sigma"].mean()
        std = posterior_draws["sigma"].std()
        print(mean, std)

        assert (mean - 2.0 * std) <= true_sigma <= (mean + 2.0 * std)
