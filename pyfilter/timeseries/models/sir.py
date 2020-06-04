from pyfilter.timeseries.diffusion import GeneralEulerMaruyama, AffineEulerMaruyama
from torch.distributions import Independent, Binomial, Normal
from pyfilter.utils import concater
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

        if initial_dist.event_shape != torch.Size([3]):
            raise NotImplementedError('Must be of size 3!')

        super().__init__(theta, initial_dist, dt=dt, prop_state=prop_state, num_steps=num_steps)

    def _propagate_u(self, x, u):
        raise ValueError(f'This method cannot sample using already defined latent variables!')


def f(x, beta, gamma, sigma):
    f1 = -beta * x[..., 0] * x[..., 1]
    f2 = x[..., 1] * (beta * x[..., 0] - gamma)
    f3 = x[..., 1] * gamma

    return concater(f1, f2, f3)


class OneFactorSIR(AffineEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Implements a SIR model where the number of sick has been replaced with the fraction of sick people of the entire
        population. Model taken from this article: https://arxiv.org/pdf/2004.06680.pdf
        :param theta: The parameters (beta, gamma, sigma)
        """

        if initial_dist.event_shape != torch.Size([3]):
            raise NotImplementedError('Must be of size 3!')

        def g(x, beta, gamma, sigma):
            g1 = -sigma * x[..., 0] * x[..., 1]
            g3 = torch.zeros_like(g1)

            return concater(g1, -g1, g3)

        inc_dist = Independent(Normal(torch.zeros(1), math.sqrt(dt) * torch.ones(1)), 1)

        super().__init__((f, g), theta, initial_dist, inc_dist, dt=dt, num_steps=num_steps)


class Mixin(object):
    def _prop(self, x, *params, dt):
        return self._helper(x, self.increment_dist.sample(x.shape[:-1]))

    def _helper(self, x, u):
        f_ = self.f(x, *self.theta_vals) * self._dt
        g = self.g(x, *self.theta_vals)

        return x + f_ + torch.matmul(g, u.unsqueeze(-1)).squeeze(-1)

    def _propagate_u(self, x, u):
        for i in range(self._ns):
            x = self._helper(x, u)

        return x


class TwoFactorSIR(Mixin, AffineEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Similar as `OneFactorFractionalStochasticSIR`, but we now have two sources of randomness originating from shocks
        to both paramters `beta` and `gamma`.
        :param theta: The parameters (beta, gamma, sigma, eta)
        """

        if initial_dist.event_shape != torch.Size([3]):
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


class ThreeFactorSIRD(Mixin, AffineEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Similar as `TwoFactorSIR`, but we now have three sources of randomness, as well as incorporating death rates.
        :param theta: The parameters (beta, gamma, alpha, rho, sigma, eta, nu)
        """

        if initial_dist.event_shape != torch.Size([4]):
            raise NotImplementedError('Must be of size 4!')

        def f_(x, beta, gamma, alpha, rho, sigma, eps, nu):
            s = -beta * x[..., 0] * x[..., 1]
            r = (1 - alpha) * gamma * x[..., 1]
            i = -s - r - alpha * rho * x[..., 1]
            d = alpha * rho * x[..., 1]

            return concater(s, i, r, d)

        def g(x, beta, gamma, alpha, rho, sigma, eps, nu):
            s = torch.zeros((*x.shape[:-1], 4, 3), device=x.device)

            s[..., 0, 0] = -sigma * x[..., 0] * x[..., 1]
            s[..., 1, 0] = -s[..., 0, 0]
            s[..., 1, 1] = -eps * (1 - alpha) * x[..., 1]
            s[..., 1, 2] = -alpha * nu * x[..., 1]
            s[..., 2, 1] = -s[..., 1, 1]
            s[..., 3, 2] = -s[..., 1, 2]

            return s

        inc_dist = Independent(Normal(torch.zeros(3), math.sqrt(dt) * torch.ones(3)), 1)

        super().__init__((f_, g), theta, initial_dist, inc_dist, dt=dt, num_steps=num_steps)


class TwoFactorSEIRD(Mixin, AffineEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Implements a two factor stochastic SEIRD model, inspired by the blog:
            https://towardsdatascience.com/infectious-disease-modelling-beyond-the-basic-sir-model-216369c584c4
        and models above.
        :param theta: The parameters of the model. Corresponds to (beta, gamma, delta, alpha, rho, sigma, eta)
        """

        def f(x, beta, gamma, delta, alpha, rho, sigma, eps):
            s = -beta * x[..., 0] * x[..., 2]
            e = -s - delta * x[..., 1]
            r = (1 - alpha) * gamma * x[..., 2]
            i = delta * x[..., 1] - r - alpha * rho * x[..., 2]
            d = alpha * rho * x[..., 2]

            return concater(s, e, i, r, d)

        def g(x, beta, gamma, delta, alpha, rho, sigma, eps):
            s = torch.zeros((*x.shape[:-1], 5, 2), device=x.device)

            s[..., 0, 0] = -sigma * x[..., 0] * x[..., 2]
            s[..., 1, 0] = -s[..., 0, 0]
            s[..., 3, 1] = eps * (1 - alpha) * x[..., 2]
            s[..., 2, 1] = -s[..., 3, 1] - alpha * eps * x[..., 2]
            s[..., 4, 1] = alpha * eps * x[..., 2]

            return s

        if initial_dist.event_shape != torch.Size([5]):
            raise NotImplementedError('Must be of size 5!')

        inc_dist = Independent(Normal(torch.zeros(2), math.sqrt(dt) * torch.ones(2)), 1)

        super().__init__((f, g), theta, initial_dist, inc_dist, dt, num_steps=num_steps)