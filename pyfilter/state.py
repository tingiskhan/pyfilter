from torch.nn import Module
from .container import TensorTupleMixin


class BaseState(Module):
    """
    Base class for state like objects.
    """


class StateWithTensorTuples(TensorTupleMixin, BaseState):
    """
    States with tensor tuples require some special handling, they should inherit from this base class instead.
    """
