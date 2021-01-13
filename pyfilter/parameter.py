from torch.nn import Parameter
import torch


class ExtendedParameter(Parameter):
    def sample_(self, prior, shape: torch.Size = None):
        self.data = prior.build_distribution().sample(shape or ())

    def update_values(self, x: torch.Tensor, prior, constrained=True):
        value = x if constrained else prior.get_constrained(x)
        support = prior().support.check(value)

        if not support.all():
            raise ValueError("Some of the values were out of bounds!")

        self[:] = value.view(self.shape)
