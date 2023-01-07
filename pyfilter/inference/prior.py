from copy import deepcopy
from typing import Tuple

import torch
from pyro.distributions import Distribution, TransformedDistribution
from pyro.distributions.transforms import Transform, biject_to


def verify_same_prior(x: Distribution, y: Distribution) -> bool:
    """
    Verifies that ``x`` and ``y`` are equivalent.

    Args:
        x (Distribution): prior ``x``.
        y (Distribution): prior ``y``.
    """

    if x.__class__ != y.__class__:
        return False

    for constraint in x.arg_constraints.keys():
        x_val = getattr(x, constraint)
        y_val = getattr(y, constraint)

        if (x_val != y_val).any():
            return False

    return True


class PriorMixin(object):
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

    def bijection(self) -> Transform:
        """
        Returns the bijection of the prior from unconstrained space to constrained.
        """

        return biject_to(self.support)

    def unconstrained_prior(self) -> TransformedDistribution:
        """
        Returns the unconstrained prior.

        Returns:
            TransformedDistribution: Unconstrained prior
        """

        return TransformedDistribution(self, self.bijection().inv)

    def get_unconstrained(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given samples ``x``, map the values to the unconstrained space of the bijected prior distribution.

        Args:
            x (torch.Tensor): samples to map to unconstrained space.

        Example:
            In the following example, we construct an exponential prior, sample from it, and then map to the
            unconstrained space (i.e. perform the mapping ``log``):

                >>> from pyro.distributions import Exponential
                >>> from pyfilter import inference
                >>> import torch
                >>>
                >>> exponential_prior = Exponential(rate=1.0)
                >>> samples = exponential_prior.sample(torch.Size([1000]))
                >>>
                >>> unconstrained = exponential_prior.get_unconstrained(samples)  # there should now be negative values
        """

        return self.bijection().inv(x)

    def get_constrained(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given samples ``x``, map the values to the constrained space of the original prior distribution.

        Args:
            x (torch.Tensor): samples to map to constrained space.

        Example:
            In the following example, we construct an exponential prior and a normal distribution, sample from the
            normal and then map to the constrained space (i.e. perform the mapping ``exp``):

                >>> from pyro.distributions import Normal, Exponential
                >>> from pyfilter import inference
                >>> import torch
                >>>
                >>> exponential_prior = Exponential(rate=1.0)
                >>> samples = Normal(0.0, 1.0).sample(torch.Size([1000]))
                >>>
                >>> constrained = exponential_prior.get_unconstrained(samples)  # all should be positive

        """

        return self.bijection()(x)

    def eval_prior(self, x: torch.Tensor, constrained: bool = True) -> torch.Tensor:
        """
        Evaluate the prior at the point ``x``.

        Args:
            x (torch.Tensor): point at which to evaluate the prior at. Note that it should always be the constrained values.
            constrained (bool): whether to transform ``x`` to unconstrained and then evaluate using the bijected prior.
        """

        if constrained:
            return self.log_prob(x)

        return self.unconstrained_prior().log_prob(self.get_unconstrained(x))

    def get_numel(self, constrained: bool = True):
        """
        Gets the number of elements of the prior, corresponding to ``.numel()`` of the ``.event_shape`` attribute of the
        prior.

        Args:
            constrained (bool): whether to get the number of elements of the constrained or unconstrained distribution.
        """

        return (self.event_shape if not constrained else self.unconstrained_prior().event_shape).numel()

    def get_slice_for_parameter(self, prev_index, constrained=True) -> Tuple[slice, int]:
        numel = self.get_numel(constrained)

        return slice(prev_index, prev_index + numel), numel

    def equivalent_to(self, other: object) -> bool:
        """
        Checks whether ``self`` is equivalent in distribution to ``other``.

        Args:
            other (object): distribution to check equivalency with.
        """

        if not isinstance(other, Distribution):
            return False

        return verify_same_prior(self, other)

    def copy(self) -> Distribution:
        """
        Copies the current instance.
        """

        return deepcopy(self)

    def to(self, device: torch.device) -> "PriorMixin":
        """
        Moves the distribution to specified device. Note that this is experimental and might not work for all
        distributions.

        Args:
            device (torch.device): device to move to.

        Returns:
            PriorMixin: Same distribution but where tensors are moved to cuda.
        """

        new = self._get_checked_instance(self.__class__, None)

        params = dict()
        for arg_name in self.arg_constraints.keys():
            parameter = getattr(self, arg_name)
            if parameter is None:
                continue

            params[arg_name] = parameter.to(device)

        new.__init__(**params)

        return new

    def cuda(self):
        """
        Moves to cuda.
        """

        return self.to("cuda:0")


applied_patches = []


# Solution found here: https://stackoverflow.com/questions/18466214/should-a-plugin-adding-new-instance-methods-monkey-patch-or-subclass-mixin-and-r
def patch(sub_cls, cls):
    """
    Function for patching an existing class with methods.

    Args:
        sub_cls (_type_): sub class to patch `cls` with.
    """

    if sub_cls in applied_patches:
        return

    for methodname in sub_cls.__dict__:
        if methodname.startswith("_") or hasattr(cls, methodname):
            continue

        method = getattr(sub_cls, methodname)
        method = get_raw_method(method)
        setattr(cls, methodname, method)

    applied_patches.append(sub_cls)


def get_raw_method(method):
    return method


patch(PriorMixin, Distribution)
