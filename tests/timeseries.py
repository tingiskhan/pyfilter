import pytest
from pyfilter import timeseries as ts, distributions as dists
import torch
from pyfilter.distributions import DistributionWrapper, Prior
from torch.distributions import Normal, Exponential
from math import sqrt


def f(x, kappa, sigma):
    return -kappa * x.values


def g(x, kappa, sigma):
    return sigma


@pytest.fixture()
def joint():
    first = ts.models.AR(0.0, 0.99, 0.05)

    scale = torch.ones(2)
    second = ts.LinearModel(
        torch.eye(2),
        0.05 * scale,
        DistributionWrapper(Normal, loc=0.0 * scale, scale=scale, reinterpreted_batch_ndims=1)
    )

    return (ts.AffineJointStochasticProcess(first=first, second=second),)


@pytest.fixture()
def chained():
    first = ts.models.AR(0.0, 0.99, 0.05)
    second = ts.LinearModel(torch.tensor([1.0, 1.0]), 0.05, DistributionWrapper(Normal, loc=0.0, scale=1.0))

    return (ts.AffineChainedStochasticProcess(first=first, second=second),)


@pytest.fixture
def custom_models(joint, chained):
    normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)

    dt = 0.05
    sde_normal = DistributionWrapper(Normal, loc=0.0, scale=sqrt(dt))

    reversion_params = (0.01, 0.05)

    return joint + chained + (
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
        ts.models.UCSV(0.01, torch.tensor([0.0, 1.0])),
        ts.models.SmoothLinearTrend(0.01, 0.0)
    )


@pytest.fixture
def proc():
    dim = 4
    priors = (
        Prior(Exponential, rate=5.0 * torch.ones(dim), reinterpreted_batch_ndims=1),
        Prior(Normal, loc=torch.zeros(dim), scale=torch.ones(dim), reinterpreted_batch_ndims=1),
        Prior(Exponential, rate=5.0 * torch.ones(dim), reinterpreted_batch_ndims=1)
    )

    return ts.models.OrnsteinUhlenbeck(*priors)


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

            is_sde = isinstance(m, (ts.StochasticDifferentialEquation, ts.OneStepEulerMaruyma))
            assert x.time_index == (num_steps * (1 if not is_sde else m.dt))

    def test_concat_parameters(self, proc: ts.AffineProcess):
        for sample_shape in (torch.Size([1]), torch.Size([100, 10, 2])):
            proc.sample_params(sample_shape)

            x = proc.concat_parameters(constrained=True, flatten=True)
            assert x.shape == torch.Size([sample_shape.numel(), 12])

            x = proc.concat_parameters(constrained=True, flatten=False)
            assert x.shape == torch.Size([*sample_shape, 12])

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
            assert x.shape == torch.Size([sample_shape.numel(), 13])

            x = ssm.concat_parameters(constrained=True, flatten=False)
            assert x.shape == torch.Size([*sample_shape, 13])

    def test_parameter_from_tensor_ssm(self, ssm):
        for sample_shape in (torch.Size([1]), torch.Size([100, 10, 2])):
            ssm.sample_params(sample_shape)

            for flatten in (True, False):
                x = ssm.concat_parameters(constrained=True, flatten=flatten)
                y = x + 1

                ssm.update_parameters_from_tensor(y, constrained=True)

                assert (y == ssm.concat_parameters(constrained=True, flatten=flatten)).all()

    def test_exogenous_variables(self):
        reversion_params = 0.0, 0.0
        normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)

        def _f(x, kappa, sigma):
            return x.exog + f(x, kappa, sigma)

        exog = torch.arange(0, self.timesteps)
        model = ts.AffineProcess((_f, g), reversion_params, normal, normal, exog=exog)

        x = model.sample_path(self.timesteps)

        assert (x.shape[0] == self.timesteps) and (x - exog)[1:].abs().max() == 1.0

        x = model.initial_sample()
        for i in range(self.timesteps):
            x = model.propagate(x)

        with pytest.raises(IndexError):
            x = model.propagate(x)

        model.append_exog(exog[-1] + 1)
        x = model.propagate(x)


@pytest.fixture()
def joint_state():
    state_1 = ts.NewState(0.0, Normal(0.0, 1.0))
    state_2 = ts.NewState(0.0, Normal(0.0, 1.0))

    return ts.JointState(state_1, state_2)


class TestState(object):
    def test_joint_state_slicing(self, joint_state):
        assert isinstance(joint_state[0], ts.NewState) and isinstance(joint_state[0].dist, Normal)

        assert isinstance(joint_state[:2], ts.JointState) and isinstance(joint_state[:2].dist, dists.JointDistribution)
        assert isinstance(joint_state[:1], ts.NewState) and isinstance(joint_state[1:], ts.NewState)

        with pytest.raises(ValueError):
            joint_state[(0, 1)]