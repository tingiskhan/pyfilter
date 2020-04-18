from .diffusion import GeneralEulerMaruyama, AffineEulerMaruyama
from torch.distributions import Independent, Binomial, Normal
from ..utils import concater
import torch
import math


def _f(x, beta, gamma, eta, dt):
    lam = beta * (x[..., 1] + eta) / x.sum(-1)

    i_f = 1. - (-lam * dt).exp()
    r_f = 1. - (-gamma * dt).exp()

    return concater(i_f, r_f)


def prop_state(x, beta, gamma, eta, dt):
    f = _f(x, beta, gamma, eta, dt)

    bins = Independent(Binomial(x[..., :-1], f), 1)
    samp = bins.sample()

    s = x[..., 0] - samp[..., 0]
    i = x[..., 1] + samp[..., 0] - samp[..., 1]
    r = x[..., 2] + samp[..., 1]

    return concater(s, i, r)


class StochasticSIR(GeneralEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Implements the stochastic SIR model.
        """

        if not initial_dist.mean.shape == torch.Size([3]):
            raise NotImplementedError('Must be of size 3!')

        super().__init__(theta, initial_dist, dt=dt, prop_state=prop_state, num_steps=num_steps)

    def _propagate_u(self, x, u):
        raise ValueError(f'This method cannot sample using already defined latent variables!')


def f(x, beta, gamma, sigma):
    f1 = -beta * x[..., 0] * x[..., 1]
    f2 = x[..., 1] * (beta * x[..., 0] - gamma)
    f3 = x[..., 1] * gamma

    return concater(f1, f2, f3)


class OneFactorFractionalStochasticSIR(AffineEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Implements a SIR model where the number of sick has been replaced with the fraction of sick people of the entire
        population. Model taken from this article: https://arxiv.org/pdf/2004.06680.pdf
        :param theta: The parameters (beta, gamma, sigma)
        """

        if not initial_dist.mean.shape == torch.Size([3]):
            raise NotImplementedError('Must be of size 3!')

        def g(x, beta, gamma, sigma):
            g1 = -sigma * x[..., 0] * x[..., 1]
            g3 = torch.zeros_like(g1)

            return concater(g1, -g1, g3)

        inc_dist = Independent(Normal(torch.zeros(1), math.sqrt(dt) * torch.ones(1)), 1)

        super().__init__((f, g), theta, initial_dist, inc_dist, dt=dt, num_steps=num_steps)


class TwoFactorFractionalStochasticSIR(AffineEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Similar as `OneFactorFractionalStochasticSIR`, but we now have two sources of randomness originating from shocks
        to both paramters `beta` and `gamma`.
        :param theta: The parameters (beta, gamma, sigma, eta)
        """

        if not initial_dist.mean.shape == torch.Size([3]):
            raise NotImplementedError('Must be of size 3!')

        def g(x, gamma, beta, sigma, eps):
            s = torch.zeros((*x.shape[:-1], 3, 2), device=x.device)

            s[..., 0, 0] = -sigma * x[..., 0] * x[..., 1]
            s[..., 1, 0] = -s[..., 0, 0]
            s[..., 1, 1] = -eps * x[..., 1]
            s[..., 2, 1] = -s[..., 1, 1]

            return s

        f_ = lambda u, beta, gamma, sigma, eps: f(u, beta, gamma, sigma)
        inc_dist = Independent(Normal(torch.zeros(2), math.sqrt(dt) * torch.ones(2)), 1)

        super().__init__((f_, g), theta, initial_dist, inc_dist, dt=dt, num_steps=num_steps)

    def _prop(self, x, *params, dt):
        f_ = self.f(x, *params) * dt
        g = self.g(x, *params)

        return x + f_ + torch.matmul(g, self.increment_dist.sample(x.shape[:-1]))

    def _propagate_u(self, x, u):
        for i in range(self._ns):
            f_ = self.f(x, *self.theta_vals) * self._dt
            g = self.g(x, *self.theta_vals)

            x += f_ + torch.matmul(g, u)

        return x
