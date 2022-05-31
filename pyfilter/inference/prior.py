from torch.distributions import TransformedDistribution, biject_to, Transform
import torch
from typing import Tuple
from stochproc.distributions.typing import HyperParameter, DistributionOrBuilder
from stochproc.distributions.base import _DistributionModule


class Prior(_DistributionModule):
    """
    Class representing a Bayesian prior on a parameter.

    Examples:
        The following example defines a prior using a normal distribution:

            >>> from torch.distributions import Normal
            >>> from pyfilter.inference.prior import Prior
            >>>
            >>> normal_prior = Prior(Normal, loc=0.0, scale=1.0)
            >>> normal_dist = normal_prior.build_distribution()

    """

    def __init__(
        self, distribution: DistributionOrBuilder, reinterpreted_batch_ndims=None, **parameters: HyperParameter
    ):
        """
        Initializes the :class:`Prior` class.

        Args:
            distribution: distribution of the prior.
            parameters: parameters of the distribution.
        """

        super().__init__(distribution, reinterpreted_batch_ndims=reinterpreted_batch_ndims)

        for k, v in parameters.items():
            self.register_buffer(k, v if isinstance(v, torch.Tensor) else torch.tensor(v))

    @property
    def device(self):
        return next(self.buffers()).device

    @property
    def bijection(self) -> Transform:
        """
        Returns the bijection of the prior from unconstrained space to constrained.
        """

        return biject_to(self().support)

    @property
    def unconstrained_prior(self) -> TransformedDistribution:
        return TransformedDistribution(self(), self.bijection.inv)

    def get_unconstrained(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given samples ``x``, map the values to the unconstrained space of the bijected prior distribution.

        Args:
            x: the samples to map to unconstrained space.

        Example:
            In the following example, we construct an exponential prior, sample from it, and then map to the
            unconstrained space (i.e. perform the mapping ``log``):

                >>> from torch.distributions import Exponential
                >>> from pyfilter.inference.prior import Prior
                >>>
                >>> exponential_prior = Prior(Exponential, rate=1.0)
                >>> samples = exponential_prior.build_distribution().sample(torch.Size([1000]))
                >>>
                >>> unconstrained = exponential_prior.get_unconstrained(samples)  # there should now be negative values
        """

        return self.bijection.inv(x)

    def get_constrained(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given samples ``x``, map the values to the constrained space of the original prior distribution.

        Args:
            x: the samples to map to constrained space.

        Example:
            In the following example, we construct an exponential prior and a normal distribution, sample from the
            normal and then map to the constrained space (i.e. perform the mapping ``exp``):

                >>> from torch.distributions import Normal, Exponential
                >>> from stochproc.distributions import Prior
                >>>
                >>> exponential_prior = Prior(Exponential, rate=1.0)
                >>> samples = Normal(0.0, 1.0).sample(torch.Size([1000]))
                >>>
                >>> constrained = exponential_prior.get_unconstrained(samples)  # all should be positive

        """

        return self.bijection(x)

    def eval_prior(self, x: torch.Tensor, constrained=True) -> torch.Tensor:
        """
        Evaluate the prior at the point ``x``.

        Args:
            x: the point at which to evaluate the prior at. Note that it should always be the constrained values.
            constrained: whether to transform ``x`` to unconstrained and then evaluate using the bijected prior.
        """

        if constrained:
            return self().log_prob(x)

        return self.unconstrained_prior.log_prob(self.get_unconstrained(x))

    def get_numel(self, constrained=True):
        """
        Gets the number of elements of the prior, corresponding to ``.numel()`` of the ``.event_shape`` attribute of the
        prior.

        Args:
            constrained: whether to get the number of elements of the constrained or unconstrained distribution.
        """

        return (self().event_shape if not constrained else self.unconstrained_prior.event_shape).numel()

    def _get_parameters(self):
        return self._buffers

    def get_slice_for_parameter(self, prev_index, constrained=True) -> Tuple[slice, int]:
        numel = self.get_numel(constrained)

        return slice(prev_index, prev_index + numel), numel
