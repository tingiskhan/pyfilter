from typing import Dict
import torch
from torch.distributions import Distribution
from . import NewState
from .stochasticprocess import StochasticProcess
from ..distributions import JointDistribution


class JointProcess(StochasticProcess):
    """
    Defines a joint stochastic process that comprises other stochastic processes.
    """

    def __init__(self, **processes: Dict[str, StochasticProcess]):
        super().__init__(initial_dist=None)

        for name, proc in processes.items():
            self.add_module(name, proc)

    def initial_dist(self) -> Distribution:
        return JointDistribution()

    def initial_sample(self, shape=None) -> NewState:
        pass

    def build_density(self, x: NewState) -> Distribution:
        pass

    def propagate_conditional(self, x: NewState, u: torch.Tensor, parameters=None, time_increment=1.0) -> NewState:
        pass