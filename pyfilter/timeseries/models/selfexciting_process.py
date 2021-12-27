import torch

from pyro.distributions import Delta
from torch.distributions import Poisson
from torch.distributions import Distribution
from ...timeseries import StochasticDifferentialEquation, NewState
from ...distributions import DistributionWrapper, JointDistribution
from ...distributions.exponentials import DoubleExponential


class LambdaProcess(StochasticDifferentialEquation):

    def __init__(self, alpha_, xi_, eta_, p, rho_minus, rho_plus, **kwargs):
        super().__init__(
            (alpha_, xi_, eta_),
            **kwargs
        )

        def _de(p_, rho_plus_, rho_minus_, **kwargs_):
            return DoubleExponential(p=p_, rho_plus=rho_plus_, rho_minus=-rho_minus_, **kwargs_)

        self.de = DistributionWrapper(_de, p_=p, rho_plus_=rho_plus, rho_minus_=rho_minus)

    def det_func(self, x, alpha, xi, eta):
        """
        deterministic part defining the evolution of (\lambda_t)_t process
        """
        return alpha * (xi - x)

    def stoch_func(self, x, alpha, xi, eta):
        """
        stochastic part defining the evolution of (\lambda_t)_t process
        """
        return eta

    def build_density(self, x: NewState) -> Distribution:
        """
        Joint density for the realizations of \lambda_t, dN_t, \lambda_s, q
        """
        alpha_, xi_, eta_ = self.functional_parameters()
        lambda_s = x.values[..., 0]

        dN_t = Poisson(rate=lambda_s * self.dt, validate_args=False).sample()
        de = self.de.build_distribution().expand(lambda_s.shape)
        q = de.sample()

        deterministic = self.det_func(lambda_s, alpha_, xi_, eta_) * self.dt

        diffusion = self.stoch_func(lambda_s, alpha_, xi_, eta_) * q.abs() * dN_t
        lambda_t = (lambda_s + deterministic + diffusion).clip(0.0, float("inf"))

        lambda_t[~torch.isfinite(lambda_t)] = lambda_s.max()  # 0.0

        return JointDistribution(Delta(lambda_t), Delta(dN_t), Delta(lambda_s), Delta(q))
