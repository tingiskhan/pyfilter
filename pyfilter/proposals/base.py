from ..timeseries.model import StateSpaceModel
from ..utils import choose
from torch.distributions import MultivariateNormal, Distribution, TransformedDistribution, AffineTransform, Independent
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
        return self._kernel

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
        Resamples the proposal
        :param inds: The indices to resample
        :type inds: torch.Tensor
        :return: Self
        :rtype: Proposal
        """

        # TODO: IMPROVE THIS

        # ===== If transformed distribution we resample everything ===== #
        if isinstance(self._kernel, TransformedDistribution):
            for at in (k for k in self._kernel.transforms if isinstance(k, AffineTransform)):
                at.loc = choose(at.loc, inds)

                if at.scale.shape == inds.shape:
                    at.scale = choose(at.scale, inds)

            return self

        # ===== Else, we just resample ===== #
        if isinstance(self._kernel, Independent):
            self._kernel.base_dist.loc = choose(self._kernel.base_dist.loc, inds)
            self._kernel.base_dist.scale = choose(self._kernel.base_dist.scale, inds)
        elif isinstance(self._kernel, MultivariateNormal):
            self._kernel.loc = choose(self._kernel.loc, inds)
            self._kernel.scale_tril = choose(self._kernel.scale_tril, inds)
        else:
            self._kernel.loc = choose(self._kernel.loc, inds)
            self._kernel.scale = choose(self._kernel.scale, inds)

        return self

    def pre_weight(self, y):
        """
        Pre-weights the old sample x. Used in the APF. Defaults to using the mean of the constructed proposal.
        Note that you should call `construct` prior to this function.
        :param y: The observation
        :type y: torch.Tensor|float
        :return: The weight
        :rtype: torch.Tensor
        """

        if not isinstance(self._kernel, (TransformedDistribution, Independent)):
            return self._model.log_prob(y, self._kernel.loc)
        elif isinstance(self._kernel, Independent):
            return self._model.log_prob(y, self._kernel.base_dist.loc)

        # TODO: Not entirely sure about this, but think this is the case
        at = next(k for k in self._kernel.transforms if isinstance(k, AffineTransform))

        return self._model.log_prob(y, at.loc)