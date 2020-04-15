from .diffusion import GeneralEulerMaruyama
from torch.distributions import Independent, Binomial
from ..utils import concater
import torch


def _f(x, beta, gamma, eta, dt):
    lam = beta * (x[1] + eta) / x.sum(-1)

    i_f = 1. - (-lam * dt).exp()
    r_f = 1. - (-gamma * dt).exp()

    return concater(i_f, r_f)


def prop_state(x, beta, gamma, eta, dt):
    f = _f(x, beta, gamma, eta, dt)

    bins = Independent(Binomial(x[:-1], f), 1)
    samp = bins.sample()

    s = x[0] - samp[..., 0]
    i = x[1] + samp[..., 0] - samp[..., 1]
    r = x[2] + samp[..., 1]

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


def _prop_state(x, beta, gamma, sigma, dt):
    # ===== Drift ===== #
    cross = x[0] * x[1]

    f1 = -beta * cross * dt
    f2 = x[1] * (beta * x[0] - gamma) * dt
    f3 = x[1] * gamma * dt

    f = concater(f1, f2, f3)

    # ===== Diffusion ===== #
    sqdt = dt.sqrt()

    g1 = -sigma * cross * sqdt
    g2 = sigma * cross * sqdt
    g3 = torch.zeros_like(f1)

    g = concater(g1, g2, g3)

    # ===== Sample ===== #
    w = torch.empty(x.shape[:-1], device=x.device).normal_()
    if x.dim() > 1:
        w.unsqueeze_(-1)

    return x + f + g * w


class FractionalStochasticSIR(GeneralEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Implements a SIR model where the number of sick has been replaced with the fraction of sick people of the entire
        population.
        :param theta: The parameters (beta, gamma, sigma)
        """

        if not initial_dist.mean.shape == torch.Size([3]):
            raise NotImplementedError('Must be of size 3!')

        super().__init__(theta, initial_dist, dt=dt, prop_state=_prop_state, num_steps=num_steps)
