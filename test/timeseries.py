import unittest
from pyfilter.timeseries import AffineModel, EulerMaruyma, OrnsteinUhlenbeck, Parameter
from pyfilter.timeseries.statevariable import StateVariable
import scipy.stats as stats
import numpy as np
import torch
from torch.distributions import Normal, Exponential


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return torch.zeros_like(alpha)


def g0(alpha, sigma):
    return sigma


class Tests(unittest.TestCase):
    norm = Normal(0., 1.)

    linear = AffineModel((f0, g0), (f, g), (1., 1.), (norm, norm))

    def test_TimeseriesCreate_1D(self):

        assert self.linear.mean(torch.tensor(1.)) == 1. and self.linear.scale(torch.tensor(1.)) == 1.

    def test_Timeseries2DState(self):
        x = np.random.normal(size=500)

        assert np.allclose(self.linear.mean(torch.tensor(x)), f(x, 1, 1))

    def test_Timeseries3DState2DParam(self):
        alpha = 0.5 * torch.ones((500, 1))
        x = torch.randn(size=(500, 200))
        linear = AffineModel((f0, g0), (f, g), (alpha, 1.), (Normal(0., 1.), Normal(0., 1.)))

        assert np.allclose(linear.mean(x), f(x, alpha, 1.))

    def test_SampleInitial(self):
        x = self.linear.i_sample()

        assert isinstance(x, torch.Tensor)

    def test_Propagate(self):
        out = torch.zeros(500)

        out[0] = self.linear.i_sample()

        for i in range(1, out.shape[0]):
            out[i] = self.linear.propagate(out[i-1])

        assert not all(out == 0.)

    def test_SampleTrajectory1D(self):
        sample = self.linear.sample(500)

        assert sample.shape[0] == 500

    def test_SampleTrajectory2D(self):
        sample = self.linear.sample(500, 250)

        assert sample.shape == (500, 250)

    def test_Weight1D(self):
        sample = self.linear.i_sample()

        obs = 0.

        assert np.allclose(self.linear.weight(obs, sample).numpy(), stats.norm.logpdf(obs, loc=sample, scale=1))

    def test_Weight2D(self):
        sample = self.linear.i_sample(shape=(500, 200))

        obs = 1

        assert np.allclose(self.linear.weight(obs, sample), stats.norm.logpdf(obs, loc=sample, scale=1))

    def test_EulerMaruyama(self):
        zero = torch.tensor(0.)
        one = torch.tensor(1.)

        mod = EulerMaruyma((lambda: zero, lambda: one), (lambda u: zero, lambda u: one), (), ndim=1)

        samples = mod.sample(30)

        assert samples.shape == (30,)

    def test_OrnsteinUhlenbeck(self):
        mod = OrnsteinUhlenbeck(0.05, 1, 0.15)

        x = mod.sample(300)

        assert x.shape == (300,)

    def test_StateVariable(self):
        # ===== Emulate last value ===== #
        rands = torch.empty((300, 3)).normal_()
        rands.requires_grad_(True)

        # ===== Pass through function ===== #
        sv = StateVariable(rands)
        agg = sv.sum(-1)

        # ===== Get gradient ===== #
        agg.backward(torch.ones_like(agg))

        assert rands.grad is not None

    def test_Parameter(self):
        # ===== Start stuff ===== #
        param = Parameter(Normal(0., 1.))
        param.sample_(1000)

        assert param.shape == torch.Size([1000])

        # ===== Construct view ===== #
        view = param.view(1000, 1)

        # ===== Change values ===== #
        param.values = torch.empty_like(param).normal_()

        assert (view[:, 0] == param).all()

        # ===== Have in tuple ===== #
        vals = (param.view(1000, 1, 1),)

        param.values = torch.empty_like(param).normal_()

        assert (vals[0][:, 0, 0] == param).all()

        # ===== Set t_values ===== #
        view = param.view(1000, 1)

        param.t_values = torch.empty_like(param).normal_()

        assert (view[:, 0] == param.t_values).all()

        # ===== Check we cannot set different shape ===== #
        with self.assertRaises(ValueError):
            param.values = torch.empty(1).normal_()

        # ===== Check that we cannot set out of bounds values for parameter ===== #
        positive = Parameter(Exponential(1.))
        positive.sample_(1)

        with self.assertRaises(ValueError):
            positive.values = -torch.empty_like(positive).normal_().abs()

        # ===== Check that we can set transformed values ===== #
        values = torch.empty_like(positive).normal_()
        positive.t_values = values

        assert (positive == positive.bijection(values)).all()