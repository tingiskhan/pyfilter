from torch.nn import Parameter
import torch
from ..distributions.prior import Prior


class RegisterParameterAndPriorMixin(object):
    """
    Mixin for registering parameters or priors on module.
    """

    def _register_parameter_or_prior(self, name: str, p):
        if isinstance(p, Prior):
            self.register_prior(name, p)
        elif isinstance(p, Parameter):
            self.register_parameter(name, p)
        else:
            self.register_buffer(name, p if (isinstance(p, torch.Tensor) or p is None) else torch.tensor(p))