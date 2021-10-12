from torch.nn import Parameter
import torch


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
        support = prior().support.check(value)

        if not support.all():
            raise ValueError("Some of the values were out of bounds!")

        self[:] = value.view(self.shape)
