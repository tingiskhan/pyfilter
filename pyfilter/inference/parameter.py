from collections import OrderedDict

import torch
from torch.nn import Parameter
from torch.utils.weak import WeakTensorKeyDictionary
from threading import local

from .prior import PriorMixin


def _rebuild_parameter(data, requires_grad, backward_hooks, name, context):
    param = PriorBoundParameter(data, requires_grad)
    param._backward_hooks = backward_hooks

    param.set_name(name)
    param.set_context(context)

    return param


_LOCALS = local()
_LOCALS.context_map = WeakTensorKeyDictionary()


class PriorBoundParameter(Parameter):
    """
    Extends :class:`torch.nn.Parameter` by adding helper methods relating to sampling and updating values from
    its bound prior.
    """

    _name: str = None

    @property
    def _context(self) -> "InferenceContext":
        return _LOCALS.context_map[self]

    def set_name(self, name: str):
        """
        Sets the name of parameter.

        Args:
            name (str): name for parameter.
        """

        self._name = name

    def set_context(self, context: "InferenceContext"):  # noqa: F821
        """
        Sets the context of the parameter.

        Args:
            context (InferenceContext): the context.
        """

        _LOCALS.context_map[self] = context

    @property
    def prior(self) -> PriorMixin:
        """
        The prior of the parameter.
        """

        return self._context.get_prior(self._name)

    def sample_(self, shape: torch.Size = torch.Size([])):
        """
        Given a prior, sample from it in-place.

        Args:
            shape (torch.Size): shape of samples.
        """

        self.copy_(self.prior.sample(shape))

    def update_values_(self, x: torch.Tensor, constrained=True):
        """
        Update the values of self with those of ``x`` in-place.

        Args:
            x (torch.Tensor): values to update self with.
            constrained (bool): whether the values ``x`` are constrained or not.
        """

        value = x if constrained else self.prior.get_constrained(x)

        # We only the support if we're considering constrained parameters as the unconstrained by definition are fine
        if constrained:
            support = self.prior.support.check(value)

            if not support.all():
                raise ValueError("Some of the values were out of bounds!")

        # Tries to set to self if congruent, else reshapes
        self.copy_(value.view(self.shape))

    def get_unconstrained(self) -> torch.Tensor:
        """
        Returns the unconstrained version of the parameter.
        """

        return self.prior.get_unconstrained(self)

    def eval_prior(self, constrained=True):
        """
        Evaluates the priors.

        Args:
            constrained (bool): whether to evaluate the constrained parameters.
        """

        if constrained:
            return self.prior.log_prob(self)

        return self.prior.unconstrained_prior().log_prob(self.get_unconstrained())

    # NB: Same as torch but we replace the `_rebuild_parameter` with our custom one.
    def __reduce_ex__(self, proto):
        return (_rebuild_parameter, (self.data, self.requires_grad, OrderedDict(), self._name, self._context))

    def __repr__(self):
        return f"PriorBoundParameter containing:\n{super(Parameter, self).__repr__()}"

    def inverse_sample_(self, probs: torch.Tensor, constrained: bool = True):
        """
        Samples from the prior by means of inversion.

        Args:
            probs (torch.Tensor): probabilities to use when inverting.
            constrained (bool): whether to sample constrained parameters.
        """

        if constrained:
            self.copy_(self.prior.icdf(probs))
            return

        unconstrained = self.prior.unconstrained_prior()
        samples = unconstrained.icdf(probs)
        self.copy_(self.prior.get_constrained(samples))
