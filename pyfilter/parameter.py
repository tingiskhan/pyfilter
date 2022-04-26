from torch.nn import Parameter
import torch
from collections import OrderedDict


def _rebuild_parameter(data, requires_grad, backward_hooks):
    param = PriorBoundParameter(data, requires_grad)
    param._backward_hooks = backward_hooks

    return param


class PriorBoundParameter(Parameter):
    """
    Extends the ``torch.nn.Parameter`` object by adding helper methods relating to sampling and updating values from
    its bound prior.
    """

    def sample_(self, prior: "Prior", shape: torch.Size = None):
        """
        Given a prior, sample from it inplace.

        Args:
            prior: The associated prior of the parameter.
            shape: The shape of samples.
        """

        self.data = prior.build_distribution().sample(shape or ())

    def update_values(self, x: torch.Tensor, prior: "Prior", constrained=True):
        """
        Update the values of ``self`` with those of ``x``.

        Args:
            x: The values to update ``self`` with.
            prior: See ``.sample_(...)``.
            constrained: Optional parameter indicating whether the values ``x`` are constrained to the prior's original
                space.
        """

        value = x if constrained else prior.get_constrained(x)

        # We only the support if we're considering constrained parameters as the unconstrained by definition are fine
        if constrained:
            support = prior().support.check(value)

            if not support.all():
                raise ValueError("Some of the values were out of bounds!")

        # Tries to set to self if congruent, else reshapes
        self[:] = value.view(self.shape) if value.numel() == self.numel() else value

    # NB: Same as torch but we replace the `_rebuild_parameter` with our custom one.
    def __reduce_ex__(self, proto):
        return (
            _rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict())
        )

    def __repr__(self):
        return f"PriorBoundParameter containing:\n{super(Parameter, self).__repr__()}"