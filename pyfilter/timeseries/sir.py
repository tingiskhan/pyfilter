from .diffusion import GeneralEulerMaruyama
from torch.distributions import Independent, Binomial
from ..utils import concater
import torch


class StochasticSIR(GeneralEulerMaruyama):
    def __init__(self, theta, initial_dist, dt, num_steps=10):
        """
        Implements the stochastic SIR model.
        """

        if not initial_dist.mean.shape == torch.Size([3]):
            raise NotImplementedError('Must be of size 3!')

        super().__init__((self._f, lambda *args: 1.), theta, initial_dist, initial_dist, dt=dt, num_steps=num_steps)

    def _f(self, x, beta, gamma, eta):
        lam = beta * (x[1] + eta) / x.sum(-1)

        i_f = 1. - (-lam * self._dt).exp()
        r_f = 1. - (-gamma * self._dt).exp()

        return concater(i_f, r_f)

    def _prop_state(self, x):
        fracs = self._f(x, *self.theta_vals)

        bins = Independent(Binomial(x[:-1], fracs), 1)
        samp = bins.sample()

        s = x[0] - samp[..., 0]
        i = x[1] + samp[..., 0] - samp[..., 1]
        r = x[2] + samp[..., 1]

        return concater(s, i, r)

    def _propagate_u(self, x, u):
        raise ValueError(f'This method cannot sample using already defined latent variables!')