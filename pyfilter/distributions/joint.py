from typing import Optional, Any, Tuple, Union, Sequence
from torch.distributions import Distribution
import torch


class JointDistribution(Distribution):
    """
    Defines an object for combining multiple distributions by assuming independence, i.e. we define:

    .. math::
        p(x_1, x_2, ..., x_n) = p(x_1) \cdot p(x_2) ... \cdot p(x_n)

    Example:
        A basic example can be seen below, where we combine a normal and and exponential distribution:

            >>> from torch.distributions import Normal, Exponential
            >>> import torch
            >>>
            >>> distribution = JointDistribution(*[Normal(0.0, 1.0), Exponential(1.0)])
            >>> y = distribution.sample(1000)
            >>>
            >>> log_prob = distribution.log_prob(y)

    """

    arg_constraints = {}

    def __init__(self, *distributions: Distribution, indices: Sequence[Union[int, slice]] = None, **kwargs):
        """
        Initializes the joint distribution object.

        Args:
            distributions: Iterable of ``pytorch.distributions.Distribution`` objects.
            indices: Optional parameter specifying which distribution corresponds to which column in input tensors. If
                ``None``, then is inferred.
            kwargs: Key-worded arguments passed to base class.
        """

        _masks = indices or self.infer_indices(*distributions)
        event_shape = torch.Size([(_masks[-1].stop if isinstance(_masks[-1], slice) else _masks[-1] + 1)])

        batch_shape = distributions[0].batch_shape
        if any(d.batch_shape != batch_shape for d in distributions):
            raise NotImplementedError(f"All batch shapes must be congruent!")

        super(JointDistribution, self).__init__(event_shape=event_shape, batch_shape=batch_shape, **kwargs)

        if any(len(d.event_shape) > 1 for d in distributions):
            raise NotImplementedError(f"Currently cannot handle matrix valued distributions!")

        self.distributions = distributions
        self.masks = _masks

    def expand(self, batch_shape, _instance=None):
        return JointDistribution(*(d.expand(batch_shape) for d in self.distributions))

    @property
    def support(self) -> Optional[Any]:
        raise NotImplementedError()

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def variance(self):
        raise NotImplementedError()

    def cdf(self, value):
        res = 0.0
        for d, m in zip(self.distributions, self.masks):
            res *= d.cdf(value[..., m])

        return res

    def icdf(self, value):
        raise NotImplementedError()

    def enumerate_support(self, expand=True):
        raise NotImplementedError()

    def entropy(self):
        return sum(d.entropy() for d in self.distributions)

    @staticmethod
    def infer_indices(*distributions: Distribution) -> Tuple[Union[int, slice]]:
        """
        Given a sequence of ``pytorch.distributions.Distribution`` objects, this method infers the indices at which to
        slice an input tensor.

        Args:
            distributions: Sequence of ``pytorch.distributions.Distribution`` objects.

        Returns:
            A tuple containing indices and/or slices.

        Example:
            >>> from torch.distributions import Normal, Exponential
            >>> import torch
            >>>
            >>> distributions = [Normal(0.0, 1.0), Exponential(1.0)]
            >>> y = torch.cat([d.sample(1000) for d in distributions], dim=-1)
            >>>
            >>> slices = JointDistribution.infer_indices(*distributions)
            >>> log_probs = [d.log_prob(y[..., s]) for d, s in zip(distributions, slices)]

        """

        res = tuple()

        length = 0
        for i, d in enumerate(distributions):
            multi_dimensional = len(d.event_shape) > 0

            if multi_dimensional:
                size = d.event_shape[-1]
                slice_ = slice(length, size + 1)

                length += slice_.stop
            else:
                slice_ = length
                length += 1

            res += (slice_,)

        return res

    def log_prob(self, value):
        # TODO: Add check for wrong dimensions
        return sum(d.log_prob(value[..., m]) for d, m in zip(self.distributions, self.masks))

    def rsample(self, sample_shape=torch.Size()):
        res = tuple(
            d.rsample(sample_shape) if len(d.event_shape) > 0 else d.rsample(sample_shape).unsqueeze(-1)
            for d in self.distributions
        )

        return torch.cat(res, dim=-1)
