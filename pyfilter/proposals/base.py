from ..timeseries.model import StateSpaceModel
from torch.distributions import Distribution
from ..module import Module
import torch


class Proposal(Module):
    def __init__(self):
        """
        Defines a proposal object for how to draw the particles.
        """

        super().__init__()

        self._model = None      # type: StateSpaceModel
        self._kernel = None     # type: Distribution

    @property
    def kernel(self):
        return self._kernel

    def set_model(self, model: StateSpaceModel):
        self._model = model

        return self

    def construct(self, y: torch.Tensor, x: torch.Tensor):
        raise NotImplementedError()

    def draw(self, rsample=False):
        if not rsample:
            return self._kernel.sample()

        return self._kernel.rsample()

    def weight(self, y: torch.Tensor, xn: torch.Tensor, xo: torch.Tensor):
        return self._model.log_prob(y, xn) + self._model.h_weight(xn, xo) - self._kernel.log_prob(xn)

    def resample(self, inds: torch.Tensor):
        return self

    def pre_weight(self, y: torch.Tensor, x: torch.Tensor):
        """
        Pre-weights the sample, used in APF.
        :param y: The next observed value
        :param x: The previous state
        :return: The pre-weights
        """

        return self._model.log_prob(y, self._model.hidden.prop_apf(x))