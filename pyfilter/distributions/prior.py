from torch.distributions import TransformedDistribution, biject_to, Transform
import torch
from typing import Tuple
from torch.distributions import Distribution
from .typing import HyperParameter, DistributionOrBuilder
from .base import DistributionBuilder


class Prior(DistributionBuilder):
    """
    Class representing a Bayesian prior on a parameter. Inherits from ``pytorch.nn.Module``.

    Examples:
        The following example defines a prior using a normal distribution:

            >>> from torch.distributions import Normal
            >>> from pyfilter.distributions import Prior
            >>>
            >>> normal_prior = Prior(Normal, loc=0.0, scale=1.0)
            >>> normal_dist = normal_prior.build_distribution()

        Note that since ``Prior`` implements ``pytorch.nn.Module``, we might as well just call ``normal_prior``.

        The next examples shows how to construct a sligthly more complicated distribution, the inverse gamma
        distribution:

            >>> from torch.distributions import Gamma, TransformedDistribution, PowerTransform
            >>> from pyfilter.distributions import Prior
            >>>
            >>> def inverse_gamma(concentration, rate, power):
            >>>     gamma = Gamma(concentration, rate)
            >>>     return TransformedDistribution(gamma, PowerTransform(power))
            >>>
            >>> inverse_gamma_prior = Prior(inverse_gamma, concentration=1.0, rate=1.0, power=-1.0)

    """

    def __init__(self, distribution: DistributionOrBuilder, reinterpreted_batch_ndims=None, **parameters: HyperParameter):
        """
        Initializes the ``Prior`` class.

        Args:
            distribution: The distribution of the prior. Can be either a type, or a callable that takes as input kwargs
                corresponding to ``parameters``.
            parameters: The parameters of the distribution.
        """

        super().__init__(distribution, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        parameters["validate_args"] = parameters.pop("validate_args", False)

        for k, v in parameters.items():
            self.register_buffer(k, v if isinstance(v, torch.Tensor) else torch.tensor(v))

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
            x: The samples to map to unconstrained space.

        Example:
            In the following example, we construct an exponential prior, sample from it, and then map to the
            unconstrained space (i.e. perform the mapping ``log``):

                >>> from torch.distributions import Exponential
                >>> from pyfilter.distributions import Prior
                >>>
                >>> exponential_prior = Prior(Exponential, rate=1.0)
                >>> samples = exponential_prior.build_distribution().sample((1000,))
                >>>
                >>> unconstrained = exponential_prior.get_unconstrained(samples)  # there should now be negative values
        """

        return self.bijection.inv(x)

    def get_constrained(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given samples ``x``, map the values to the constrained space of the original prior distribution.

        Args:
            x: The samples to map to constrained space.

        Example:
            In the following example, we construct an exponential prior and a normal distribution, sample from the
            normal and then map to the constrained space (i.e. perform the mapping ``exp``):

                >>> from torch.distributions import Normal, Exponential
                >>> from pyfilter.distributions import Prior
                >>>
                >>> exponential_prior = Prior(Exponential, rate=1.0)
                >>> samples = Normal(0.0, 1.0).sample((1000,))
                >>>
                >>> constrained = exponential_prior.get_unconstrained(samples)  # all should be positive

        """

        return self.bijection(x)

    def eval_prior(self, x: torch.Tensor, constrained=True) -> torch.Tensor:
        """
        Evaluate the prior at the point ``x``.

        Args:
            x: The point at which to evaluate the prior at. Note that it should always be the constrained values.
            constrained: Optional parameter for whether to transform ``x`` to unconstrained and then evaluate the prior
                using the bijected prior.
        """

        if constrained:
            return self().log_prob(x)

        return self.unconstrained_prior.log_prob(self.get_unconstrained(x))

    def get_numel(self, constrained=True):
        """
        Gets the number of elements of the prior, corresponding to ``.numel()`` of the ``.event_shape`` attribute of the
        prior.

        Args:
            constrained: Whether to get the number of elements of the constrained or unconstrained distribution.
        """

        return (self().event_shape if not constrained else self.unconstrained_prior.event_shape).numel()

    def _get_parameters(self):
        return self._buffers

    def get_slice_for_parameter(self, prev_index, constrained=True) -> Tuple[slice, int]:
        numel = self.get_numel(constrained)

        return slice(prev_index, prev_index + numel), numel
