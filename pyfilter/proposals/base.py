from ..timeseries.model import StateSpaceModel
from torch.distributions import Distribution
from ..module import Module


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
        :rtype: Distribution
        """

        return self._kernel

    def set_model(self, model):
        """
        Sets the model and all required attributes.
        :param model: The model to ues
        :type model: StateSpaceModel
        :return: Self
        :rtype: Proposal
        """

        self._model = model

        return self

    def construct(self, y, x):
        """
        Constructs the kernel used in `draw` and `weight`.
        :param y: The observation
        :type y: torch.Tensor
        :param x: The old state
        :type x: torch.Tensor
        :return: Self
        :rtype: Proposal
        """

        raise NotImplementedError()

    def draw(self, rsample=False):
        """
        Defines the method for drawing proposals.
        :param rsample: Whether to use `rsample` instead
        :type rsample: bool
        :rtype: torch.Tensor
        """

        if not rsample:
            return self._kernel.sample()

        return self._kernel.rsample()

    def weight(self, y, xn, xo):
        """
        Defines the method for weighting observations.
        :param y: The current observation
        :type y: torch.Tensor
        :param xn: The new state
        :type xn: torch.Tensor
        :param xo: The old state
        :type xo: torch.Tensor
        :rtype: torch.Tensor
        """

        return self._model.log_prob(y, xn) + self._model.h_weight(xn, xo) - self._kernel.log_prob(xn)

    def resample(self, inds):
        """
        Resamples the proposal. Used for proposals when there's a separate module constructing the proposal.
        :param inds: The indices to resample
        :type inds: torch.Tensor
        :return: Self
        :rtype: Proposal
        """

        return self

    def pre_weight(self, y, x):
        """
        Pre-weights the sample, used in APF.
        :param y: The next observed value
        :type y: torch.Tensor
        :param x: The previous state
        :type x: torch.Tensor
        :return: The pre-weights
        :rtype: torch.Tensor
        """

        return self._model.log_prob(y, self._model.hidden.prop_apf(x))