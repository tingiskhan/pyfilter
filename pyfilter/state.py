from torch.nn import Module
from .utils import TensorTuple


class BaseState(Module):
    """
    Base class for state like objects.
    """


class StateWithTensorTuples(BaseState):
    """
    States with tensor tuples require some special handling, they should inherit from this base class instead.
    """

    def __init__(self):
        super().__init__()

        self._register_state_dict_hook(TensorTuple.dump_hook)
        self._register_load_state_dict_pre_hook(lambda *args: TensorTuple.load_hook(self, *args))

    def _apply(self, fn):
        super(BaseState, self)._apply(fn)

        for k, v in filter(lambda kk, vv: isinstance(vv, TensorTuple), vars(self).items()):
            v.apply(fn)

        return self
