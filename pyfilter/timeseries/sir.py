from .diffusion import GeneralEulerMaruyama
from torch.distributions import Independent, Binomial
from ..utils import concater
import torch


def _f(x, beta, gamma, eta, dt):
    lam = beta * (x[1] + eta) / x.sum(-1)

    i_f = 1. - (-lam * dt).exp()
    r_f = 1. - (-gamma * dt).exp()

    return concater(i_f, r_f)


def prop_state(x, f, g):
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

        funcs = (_f, lambda *args: 1.)
        super().__init__(funcs, theta, initial_dist, initial_dist, dt=dt, prop_state=prop_state, num_steps=num_steps)

    def _propagate_u(self, x, u):
        raise ValueError(f'This method cannot sample using already defined latent variables!')