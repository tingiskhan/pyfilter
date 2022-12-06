from torch.nn import Parameter
import torch
from collections import OrderedDict

from .prior import Prior


def _rebuild_parameter(data, requires_grad, backward_hooks, name, context):
    param = PriorBoundParameter(data, requires_grad)
    param._backward_hooks = backward_hooks

    param.set_name(name)
    param.set_context(context)

    return param


class PriorBoundParameter(Parameter):
    """
    Extends :class:`torch.nn.Parameter` by adding helper methods relating to sampling and updating values from
    its bound prior.
    """

    _context: "ParameterContext" = None  # noqa: F821
    _name: str = None

    def set_name(self, name: str):
        """
        Sets the name of parameter.

        Args:
            name: name for parameter.
        """

        self._name = name

    # TODO: Should be done on __init__/__new__
    # TODO: Might have to add
    def set_context(self, context: "ParameterContext"):  # noqa: F821
        """
        Sets the context of the parameter.

        Args:
            context: the context.
        """

        self._context = context

    @property
    def prior(self) -> Prior:
        """
        The prior of the parameter.
        """

        # TODO: Far from optimal...
        prior = self._context.get_prior(self._name)
        if self.device != prior.device:
            self._context._prior_dict[self._name] = prior = prior.to(self.device)

        return prior

    def sample_(self, shape: torch.Size = torch.Size([])):
        """
        Given a prior, sample from it inplace.

        Args:
            shape: shape of samples.
        """

        self.copy_(self.prior.build_distribution().sample(shape))

    def update_values_(self, x: torch.Tensor, constrained=True):
        """
        Update the values of self with those of ``x`` inplace.

        Args:
            x: values to update self with.
            constrained: whether the values ``x`` are constrained or not.
        """

        value = x if constrained else self.prior.get_constrained(x)

        # We only the support if we're considering constrained parameters as the unconstrained by definition are fine
        if constrained:
            support = self.prior().support.check(value)

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
            constrained: whether to evaluate the constrained parameters.
        """

        if constrained:
            return self.prior.build_distribution().log_prob(self)

        return self.prior.unconstrained_prior.log_prob(self.get_unconstrained())

    # NB: Same as torch but we replace the `_rebuild_parameter` with our custom one.
    def __reduce_ex__(self, proto):
        return (_rebuild_parameter, (self.data, self.requires_grad, OrderedDict(), self._name, self._context))

    def __repr__(self):
        return f"PriorBoundParameter containing:\n{super(Parameter, self).__repr__()}"

    def inverse_sample_(self, probs: torch.Tensor, constrained: bool = True):
        r"""
        Samples from the prior by means of inversion.

        Args:
            probs: probabilities to use when inverting.
            constrained: whether to sample constrained.
        """

        if constrained:
            self.copy_(self.prior.build_distribution().icdf(probs))
            return

        unconstrained = self.prior.unconstrained_prior
        samples = unconstrained.icdf(probs)
        self.copy_(self.prior.get_constrained(samples))
