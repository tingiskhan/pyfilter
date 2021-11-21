import torch
from torch.distributions import Distribution, constraints, utils, Uniform
from ..typing import ArrayType


class AsymmetricLaplace(Distribution):
    """
    Implements the `Asymmetric Laplace` distribution, with pdf given by:
        .. math::
            f(x; m, \\lambda, \\kappa) = \\frac{\\lambda}{\\kappa + 1/\\kappa)}e^{-(x-m)\\lambda s \\kappa^s}, \n
            s = sign(x - m).

    See more information `here`_.

    .. _here: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
    """

    arg_constraints = {"scale": constraints.positive, "skew": constraints.positive, "loc": constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, scale: ArrayType, skew: ArrayType, loc: ArrayType = 0.0, **kwargs):
        """
        Initializes the ``AsymmetricLaplace`` distribution.

        Args:
            scale: The scale parameter, denoted :math:`\\lambda`.
            skew: The skew parameter, denoted :math:`\\kappa`.
            loc: Optional parameter, the location of the distribution.
        """

        self.scale, self.skew, self.loc = utils.broadcast_all(scale, skew, loc)
        super().__init__(**kwargs)

    def expand(self, batch_shape, _instance=None):
        return AsymmetricLaplace(self.scale.expand(batch_shape), self.skew.expand(batch_shape))

    @property
    def mean(self):
        return self.loc + (1 - self.skew ** 2) / self.scale / self.skew

    @property
    def variance(self):
        return (1 + self.skew ** 4) / (self.scale * self.skew) ** 2

    def rsample(self, sample_shape=torch.Size()):
        u = Uniform(-self.skew, 1.0 / self.skew, validate_args=self._validate_args).rsample(sample_shape)
        s = torch.sign(u)

        s_skew = s * self.skew ** s

        return self.loc - 1 / self.scale / s_skew * (1 - u * s_skew).log()
        
    def log_prob(self, value):
        first = self.scale.log() - (self.skew + 1 / self.skew).log()

        y = value - self.loc
        s = torch.sign(y)
        second = -y * self.scale * s * self.skew ** s

        return first + second

    def cdf(self, value):
        raise NotImplementedError()
