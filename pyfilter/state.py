from torch.nn import Module
from collections import OrderedDict
from .utils import TensorTuple


class BaseState(Module):
    """
    Base class for state like objects.
    """


class StateWithTensorTuples(BaseState):
    """
    States with tensor tuples require some special handling, they should inherit from this base class instead.
    """

    _TENSOR_TUPLE_PREFIX = "tensor_tuples"

    def __init__(self):
        super().__init__()

        self._register_state_dict_hook(self.dump_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

        self.tensor_tuples: OrderedDict[str, TensorTuple] = OrderedDict()

    def _apply(self, fn):
        super(BaseState, self)._apply(fn)

        for k, v in self.tensor_tuples.items():
            v.apply(fn)

        return self

    @staticmethod
    def dump_hook(self, state_dict, prefix, local_metadata):
        # TODO: Might have use prefix?
        for k, v in self.tensor_tuples.items():
            state_dict[f"{self._TENSOR_TUPLE_PREFIX}.{k}"] = v.tensors

    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        to_pop = list()
        # TODO: Might have use prefix?
        # TODO: Correct to only check startswith?
        for k, v in filter(lambda x: x[0].startswith(self._TENSOR_TUPLE_PREFIX), state_dict.items()):
            self.tensor_tuples[k.replace(f"{self._TENSOR_TUPLE_PREFIX}.", "")] = TensorTuple(*v)
            to_pop.append(k)

        for tp in to_pop:
            state_dict.pop(tp)