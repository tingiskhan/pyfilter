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

    def modules(self):
        return {}

    @property
    def kernel(self):
        """
        Returns the latest kernel
        """

        return self._kernel

    def set_model(self, model: StateSpaceModel):
        """
        Sets the model and all required attributes.
        :param model: The model to use
        :return: Self
        """

        self._model = model

        return self

    def construct(self, y: torch.Tensor, x: torch.Tensor):
        """
        Constructs the kernel used in `draw` and `weight`.
        :param y: The observation
        :param x: The old state
        :return: Self
        """

        raise NotImplementedError()

    def draw(self, rsample=False):
        """
        Defines the method for drawing proposals.
        :param rsample: Whether to use `rsample` instead
        """

        if not rsample:
            return self._kernel.sample()

        return self._kernel.rsample()

    def weight(self, y: torch.Tensor, xn: torch.Tensor, xo: torch.Tensor):
        """
        Defines the method for weighting observations.
        :param y: The current observation
        :param xn: The new state
        :param xo: The old state
        """

        return self._model.log_prob(y, xn) + self._model.h_weight(xn, xo) - self._kernel.log_prob(xn)

    def resample(self, inds: torch.Tensor):
        """
        Resamples the proposal. Used for proposals when there's a separate module constructing the proposal.
        :param inds: The indices to resample
        :return: Self
        """

        return self

    def pre_weight(self, y: torch.Tensor, x: torch.Tensor):
        """
        Pre-weights the sample, used in APF.
        :param y: The next observed value
        :param x: The previous state
        :return: The pre-weights
        """

        return self._model.log_prob(y, self._model.hidden.prop_apf(x))