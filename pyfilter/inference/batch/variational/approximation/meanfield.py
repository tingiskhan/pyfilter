import torch
from torch.distributions import Independent, Normal, TransformedDistribution, Distribution, LogNormal
from torch.nn import Parameter
from typing import Tuple
from .base import BaseApproximation
from ....utils import priors_from_model


class StateMeanField(BaseApproximation):
    """
    Mean field approximation for states. Assumes that the state distributions can be approximated using independent
    normal distributions parameterized using a mean and scale, i.e. in which we approximate the state distribution by
        .. math::
            p(x_0, x_1, \\dots, x_n) = \\prod_{i=0}^n \\mathcal{N}(x_i \\mid \\mu_i, \\sigma_i).
    """

    def __init__(self):
        """
        Initializes the ``StateMeanField`` class.
        """

        super().__init__()
        self._dim = None

        self.mean = None
        self.log_std = None

    def initialize(self, data, model):
        mean = torch.zeros((data.shape[0] + 1, *model.hidden.increment_dist().event_shape))
        log_std = torch.zeros_like(mean)

        self.mean = Parameter(mean, requires_grad=True)
        self.log_std = Parameter(log_std, requires_grad=True)

        self._dim = model.hidden.n_dim

        return self

    def get_approximation(self) -> Distribution:
        return Independent(Normal(self.mean, self.log_std.exp()), self._dim + 1)

    def get_inferred_states(self) -> torch.Tensor:
        """
        Returns the mean of the inferred states, which corresponds to the mean of the variational approximation.
        """

        return self.mean


class ParameterMeanField(BaseApproximation):
    """
    Mean field approximation for parameters. Assumes that the `unconstrained` parameter distributions can be
    approximated using normal distributions parameterized using a mean and scale, i.e.
        .. math::
            p(\\theta_1, \\dots, \\theta__n) = \\prod_{i=1}^n \\mathcal{N}(\\theta_i \\mid \\mu_i, \\sigma_i).

    """

    def __init__(self):
        """
        Initializes the ``ParameterMeanField`` class.
        """

        super().__init__()
        self._bijections = None
        self._mask = None

        self.mean = None
        self.log_std = None

    def get_parameters(self):
        return self.mean, self.log_std

    def initialize(self, data, model):
        self._bijections = tuple()
        self._mask = tuple()
        means = tuple()

        left = 0
        for p in priors_from_model(model):
            slc, numel = p.get_slice_for_parameter(left, False)

            val = p.bijection.inv(p().mean)
            if val.dim() == 0:
                val.unsqueeze_(-1)

            means += (val,)
            self._bijections += (p.bijection,)
            self._mask += (slc,)

            left += numel

        mean = torch.cat(means)
        log_std = torch.zeros_like(mean)

        self.mean = Parameter(mean, requires_grad=True)
        self.log_std = Parameter(log_std, requires_grad=True)

        return self

    def get_approximation(self) -> Distribution:
        return Independent(Normal(self.mean, self.log_std.exp()), 1)

    def get_transformed_dists(self) -> Tuple[Distribution, ...]:
        """
        Returns the transformed distributions of the parameter approximations from unconstrained to constrained.
        """

        res = tuple()
        for bij, msk in zip(self._bijections, self._mask):
            dist = TransformedDistribution(Normal(self.mean[msk], self.log_std[msk].exp()), bij)
            res += (dist,)

        return res
