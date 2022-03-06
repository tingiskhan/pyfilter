from torch.distributions import Poisson, constraints
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all

from pyfilter.timeseries import StochasticDifferentialEquation, NewState
from pyfilter.distributions import DistributionWrapper, JointDistribution
from numbers import Number
import torch
from pyro.distributions import Delta

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
        return alpha * (xi - x)

    def stoch_func(self, x, alpha, xi, eta):
        return eta

    def build_density(self, x: NewState) -> Distribution:
        alpha_, xi_, eta_ = self.functional_parameters()
        lambda_s = x.values[..., 0]

        dN_t = Poisson(rate=lambda_s * self.dt, validate_args=False).sample()
        de = self.de.build_distribution().expand(lambda_s.shape)
        dL_t = de.sample() * dN_t

        deterministic = self.det_func(lambda_s, alpha_, xi_, eta_) * self.dt

        diffusion = self.stoch_func(lambda_s, alpha_, xi_, eta_) * dL_t.abs() 
        lambda_t = (lambda_s + deterministic + diffusion).clip(0.0, float("inf"))

        lambda_t[~torch.isfinite(lambda_t)] = lambda_s.max()  # 0.0

        return JointDistribution(Delta(lambda_t), Delta(dN_t), Delta(lambda_s), Delta(dL_t))
