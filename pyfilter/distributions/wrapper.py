from typing import Union
from .base import DistributionBuilder
from .prior import Prior
from ..prior_module import HasPriorsModule
from .typing import DistributionOrBuilder, HyperParameter


class DistributionWrapper(DistributionBuilder, HasPriorsModule):
    """
    Implements a wrapper around ``pytorch.distributions.Distribution`` objects. It inherits from ``pytorch.nn.Module``
    in order to utilize all of the associated methods and attributes. One such is e.g. moving tensors between different
    devices.

    Example:
        >>> from torch.distributions import Normal
        >>> from pyfilter.distributions import DistributionWrapper
        >>>
        >>> wrapped_normal_cpu = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        >>> wrapped_normal_cuda = wrapped_normal_cpu.cuda()
        >>>
        >>> cpu_samples = wrapped_normal_cpu.build_distribution().sample((1000,)) # device cpu
        >>> cuda_samples = wrapped_normal_cuda.build_distribution().sample((1000,)) # device cuda
    """

    def __init__(
        self,
        base_dist: DistributionOrBuilder,
        reinterpreted_batch_ndims=None,
        **parameters: Union[HyperParameter, Prior]
    ):
        """
        Initializes the ``DistributionWrapper`` class.

        Args:
            base_dist: See the ``distribution`` of ``pyfilter.distributions.Prior``.
            parameters: See ``parameters`` of ``pyfilter.distributions.Prior``. With the addition that we can pass
                ``pyfilter.distributions.Prior`` objects as parameters.

        Example:
            In this example we'll construct a distribution wrapper around a normal distribution where the location is a
            prior:
                >>> from torch.distributions import Normal
                >>> from pyfilter.distributions import DistributionWrapper, Prior
                >>>
                >>> loc_prior = Prior(Normal, loc=0.0, scale=1.0)
                >>> wrapped_normal_with_prior = DistributionWrapper(Normal, loc=loc_prior, scale=1.0)
                >>>
                >>> wrapped_normal_with_prior.sample_params((1000,))
                >>> samples = wrapped_normal_with_prior.build_distribution().sample((1000,)) # should be 1000 x 1000
        """

        super(DistributionWrapper, self).__init__(
            base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims
        )
        parameters["validate_args"] = parameters.pop("validate_args", False)

        for k, v in parameters.items():
            self._register_parameter_or_prior(k, v)

    def _get_parameters(self):
        return self.parameters_and_buffers()
