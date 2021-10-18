import pytest
from pyfilter import timeseries as ts
import torch
from pyfilter.distributions import DistributionWrapper, Prior
from torch.distributions import Normal, Independent, Exponential
from math import sqrt


def f(x, kappa, sigma):
    return -kappa * x.values


def g(x, kappa, sigma):
    return sigma


@pytest.fixture
def custom_models():
    normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)

    dt = 0.05
    sde_normal = DistributionWrapper(Normal, loc=0.0, scale=sqrt(dt))

    reversion_params = (0.01, 0.05)

    return (
        ts.AffineProcess((f, g), reversion_params, normal, normal),
        ts.AffineEulerMaruyama((f, g), reversion_params, normal, sde_normal, dt=dt),
        ts.OneStepEulerMaruyma((f, g), reversion_params, normal, sde_normal, dt=dt),
        ts.Euler(lambda *u: f(*u, 0.0), reversion_params[:1], 5.0, dt=dt, tuning_std=1e-2),
        ts.RungeKutta(lambda *u: f(*u, 0.0), reversion_params[:1], 5.0, dt=dt, tuning_std=1e-2),
        ts.RungeKutta(lambda *u: f(*u, 0.0), reversion_params[:1], torch.tensor([5.0, 1.0]), dt=dt, tuning_std=1e-2)
    )


@pytest.fixture
def timeseries_models(custom_models):
    return custom_models + (
        ts.models.AR(0.0, 0.99, 0.05),
        ts.models.LocalLinearTrend(torch.tensor([1e-3, 1e-2])),
        ts.models.OrnsteinUhlenbeck(0.01, 0.0, 0.05),
        ts.models.OrnsteinUhlenbeck(0.01 * torch.ones(2), torch.zeros(2), 0.05 * torch.ones(2)),
        ts.models.RandomWalk(0.05),
        ts.models.RandomWalk(0.05 * torch.ones(2)),
        ts.models.Verhulst(0.01, 1.0, 0.05, 1.0),
        ts.models.SemiLocalLinearTrend(0.0, 0.99, torch.tensor([1e-3, 1e-2])),
        ts.models.UCSV(0.01, torch.tensor([0.0, 1.0])),
    )


@pytest.fixture
def proc():
    priors = (
        Prior(Normal, loc=0.0, scale=1.0),
        Prior(lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(3), scale=torch.ones(3)),
    )

    dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)
    return ts.AffineProcess((None, None), priors, dist, dist)


@pytest.fixture
def ssm(proc):
    return ts.LinearGaussianObservations(proc, torch.tensor([1.0, 0.0, 0.0, 0.0]), Prior(Exponential, rate=5.0))


class TestTimeseries(object):
    timesteps = 1000

    def test_correct_order(self):
        parameters = 0.0, 0.99, 0.05
        model = ts.models.AR(*parameters)

        for p, true_p in zip(model.functional_parameters(), parameters):
            assert p == true_p

    def test_correct_order_with_prior(self):
        parameters = 0.0, Prior(Normal, loc=0.0, scale=1.0), 0.05
        model = ts.models.AR(*parameters)

        for i, (p, true_p) in enumerate(zip(model.functional_parameters(), parameters)):
            if i == 1:
                assert (true_p is parameters[1]) and (true_p in model.priors())
            else:
                assert p == true_p

    def test_sample_path(self, timeseries_models):
        for m in timeseries_models:
            path = m.sample_path(self.timesteps)

            assert path.shape[0] == self.timesteps

    def test_sample_path_batched(self, timeseries_models):
        samples = torch.Size([10, 10])
        for m in timeseries_models:
            path = m.sample_path(self.timesteps, samples=samples)

            assert path.shape[1:3] == samples

    def test_sample_num_steps(self, timeseries_models):
        num_steps = 5

        for m in timeseries_models:
            m.num_steps = num_steps

            x = m.initial_sample()
            x = m.propagate(x)

            assert x.time_index == num_steps

    def test_concat_parameters(self, proc: ts.AffineProcess):
        for sample_shape in (torch.Size([1]), torch.Size([100, 10, 2])):
            proc.sample_params(sample_shape)

            x = proc.concat_parameters(constrained=True, flatten=True)
            assert x.shape == torch.Size([sample_shape.numel(), 4])

            x = proc.concat_parameters(constrained=True, flatten=False)
            assert x.shape == torch.Size([*sample_shape, 4])

    def test_parameter_from_tensor(self, proc: ts.AffineProcess):
        for sample_shape in (torch.Size([1]), torch.Size([100, 10, 2])):
            proc.sample_params(sample_shape)

            for flatten in (True, False):
                x = proc.concat_parameters(constrained=True, flatten=flatten)
                y = x + 1

                proc.update_parameters_from_tensor(y, constrained=True)

                assert (y == proc.concat_parameters(constrained=True, flatten=flatten)).all()

    def test_concat_parameters_ssm(self, ssm):
        for sample_shape in (torch.Size([1]), torch.Size([100, 10, 2])):
            ssm.sample_params(sample_shape)

            x = ssm.concat_parameters(constrained=True, flatten=True)
            assert x.shape == torch.Size([sample_shape.numel(), 5])

            x = ssm.concat_parameters(constrained=True, flatten=False)
            assert x.shape == torch.Size([*sample_shape, 5])

    def test_parameter_from_tensor_ssm(self, ssm):
        for sample_shape in (torch.Size([1]), torch.Size([100, 10, 2])):
            ssm.sample_params(sample_shape)

            for flatten in (True, False):
                x = ssm.concat_parameters(constrained=True, flatten=flatten)
                y = x + 1

                ssm.update_parameters_from_tensor(y, constrained=True)

                assert (y == ssm.concat_parameters(constrained=True, flatten=flatten)).all()
