from torch.nn import Parameter
import torch


# Basically same as old
def _rebuild_parameter(prior, data, requires_grad, backward_hooks):
    param = torch.nn.Parameter(prior, requires_grad)
    param.data = data
    param._backward_hooks = backward_hooks

    return param


class ExtendedParameter(Parameter):
    def sample_(self, prior, shape: torch.Size = None):
        self.data = prior.build_distribution().sample(shape or ())

    def update_values(self, x: torch.Tensor, prior, constrained=True):
        value = x if constrained else prior.get_constrained(x)
        support = prior().support.check(value)

        if not support.any():
            raise ValueError("Some of the values were out of bounds!")

        self[:] = value.view(self.shape)
