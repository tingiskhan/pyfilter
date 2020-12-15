from torch.nn import Parameter
import torch
from .distributions import Prior


class ExtendedParameter(Parameter):
    def __new__(cls, prior: Prior, requires_grad=False):
        return torch.Tensor._make_subclass(cls, torch.empty(prior().event_shape), requires_grad)

    def __init__(self, prior: Prior, **kwargs):
        super().__init__(**kwargs)
        self.prior = prior

    def sample_(self, shape: torch.Size = None):
        self.data = self.prior.build_distribution().sample(shape or ())
