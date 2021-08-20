import torch
from torch.distributions import Distribution
from torch.nn import Module
from abc import ABC
from ....utils import Process


class BaseApproximation(Module, ABC):
    """
    Base class for constructing variational approximations.
    """

    def __init__(self):
        super().__init__()

    def initialize(self, data: torch.Tensor, model: Process):
        return self

    def dist(self) -> Distribution:
        raise NotImplementedError()
