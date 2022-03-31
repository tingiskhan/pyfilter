from torch.distributions import ExponentialFamily, constraints, Exponential, TransformedDistribution, AffineTransform
from torch.distributions.utils import broadcast_all
from ..typing import NumberOrTensor
from numbers import Number
import torch


class NegativeExponential(TransformedDistribution):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::
        >>> m = NegativeExponential(-torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=-1
        tensor([ 0.1046])
    """

    def entropy(self):
        return 1.0 - self.base_dist.rate.log()

    arg_constraints = {"rate": constraints.less_than(0.0)}

    def __init__(self, rate: NumberOrTensor, **kwargs):
        """
        Initializes the ``NegativeExponential`` class.

        Args:
            rate: rate = - 1 / scale of the distribution (abs value)
        """

        neg_rate = broadcast_all(-rate)
        loc = torch.zeros_like(neg_rate)
        scale = -torch.ones_like(neg_rate)

        super().__init__(Exponential(neg_rate), AffineTransform(loc, scale), **kwargs)

    @property
    def mean(self):
        return -self.base_dist.rate.reciprocal()

    @property
    def stddev(self):
        return self.base_dist.rate.reciprocal()

    @property
    def variance(self):
        return self.base_dist.rate.pow(-2)


# TODO: Verify that the mask actually nulls out inf
class DoubleExponential(ExponentialFamily):
    r"""
    Creates a Double Exponential distribution parameterized by :attr:`rho_minus, rho_plus, p`.

    Example::
        >>> m = DoubleExponential(torch.tensor([-10, +10, 0.5]))
        >>> m.sample()  # Double Exponential distributed with pdf zeta(q) = p * rho_plus * exp(-rho_plus * q) if q > 0
        tensor([ 0.1046])
    """

    arg_constraints = {
        "rho_plus": constraints.positive,
        "rho_minus": constraints.less_than(0.),
        "p": constraints.interval(0.0, 1.0)
    }

    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.p / self.rho_plus + (1 - self.p) / self.rho_minus

    @property
    def stddev(self):
        # V(J) := E[J^2] - E[J]^2
        return (self.p * 2 * self.rho_plus.pow(-2.) + (1 - self.p) * 2 * self.rho_minus.pow(-2.) - self.mean.pow(2.)).sqrt()

    @property
    def variance(self):
        sigma = self.stddev
        return sigma.pow(2.)

    @property
    def phi_fun(self):
        # eq. 30 in Hainaut n Moraux 2016
        # \phi(1, 0) =: c
        return self.p * (1 / self.rho_plus).exp() + (1 - self.p) * (1 / self.rho_minus).exp()

    def __init__(self, rho_plus: NumberOrTensor, rho_minus: NumberOrTensor, p: NumberOrTensor, validate_args=None):
        """
        Initializes the ``DoubleExponential`` class.

        Args:
            rho_plus: rho_plus = 1 / rho_plus of the distribution of positive z
            rho_minus: rho_minus = 1 / rho_minus of the distribution of negative z
            p: p probability of getting a positive z
        """

        self.rho_plus, self.rho_minus, self.p, = broadcast_all(rho_plus, rho_minus, p)
        super(DoubleExponential, self).__init__(self.p.shape, validate_args=validate_args)
        self.c = self.phi_fun

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DoubleExponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.p = self.p.expand(batch_shape)
        new.rho_plus = self.rho_plus.expand(batch_shape)
        new.rho_minus = self.rho_minus.expand(batch_shape)

        super(DoubleExponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(size=shape, device=self.p.device)

        mask = u < self.p
        not_mask = ~mask

        x = (
            (-1.0 / self.rho_minus * (u / (1.0 - self.p))).log() * mask +
            (-1.0 / self.rho_plus * ((1.0 - u) / self.p).log()) * not_mask
        )

        return x

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        neg_mask = value <= 0.0
        log_prob = (
                ((self.p + self.rho_plus).log() - self.rho_plus * value) * (~neg_mask) +
                ((-self.rho_minus * (1 - self.p)).log() - self.rho_minus * value) * neg_mask
        )

        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        neg_mask = value <= 0.0

        pm1 = 1 - self.p
        u = (
                (pm1 + self.p * (1 - (-self.rho_plus * value).exp())) * (~neg_mask) +
                (pm1 * (1 - (-self.rho_minus * value).exp())) * neg_mask
        )

        return u

    # TODO: Does this really work?
    def icdf(self, value):
        n = value.size()
        u = torch.zeros(n)
        x = torch.zeros(n)
        for i in range(0, n.__getitem__(0)):
            if u[i] <= self.p:
                x[i] = - 1 / self.rho_minus * (u[i] / (1 - self.p)).log()
            else:
                x[i] = - 1 / self.rho_plus * ((1 - u[i]) / self.p).log()
        return x

    def entropy(self):
        return (1.0 - self.p) * (1.0 - torch.log(-(1.0 - self.p) * self.rho_minus)) + self.p * (1.0 - torch.log(self.p * self.rho_plus))

    @property
    def _natural_params(self):
        return self.rho_plus, self.rho_minus, self.p

    def _log_normalizer(self, x):
        return -torch.log(-x)
