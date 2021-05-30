from typing import Optional, Any, Dict
from torch.distributions import Distribution, constraints
from torch.distributions.utils import _sum_rightmost
import torch


class JointDistribution(Distribution):
    """
    Defines an object for combining multiple distributions into one. Assumes independence.
    """

    def __init__(self, *distributions: Distribution, **kwargs):
        super().__init__(**kwargs)

        if any(len(d.event_shape) > 1 for d in distributions):
            raise NotImplementedError(f"Currently cannot handle matrix valued distributions!")

        self.distributions = distributions
        self.masks = self.get_mask(*distributions)

    def expand(self, batch_shape, _instance=None):
        return JointDistribution(*(d.expand(batch_shape) for d in self.distributions))

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        raise NotImplementedError()

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
    def get_mask(*distributions: Distribution):
        res = tuple()

        length = 0
        for i, d in enumerate(distributions):
            multi_dimensional = len(d.event_shape) > 0

            if multi_dimensional:
                size = d.event_shape[-1]
                slice_ = slice(length, size)

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
