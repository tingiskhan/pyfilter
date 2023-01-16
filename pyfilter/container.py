import itertools
from collections import OrderedDict, deque
from typing import Deque, Dict, Iterable, Iterator, Tuple, Union

import torch

BoolOrInt = Union[int, bool]


def make_dequeue(maxlen: BoolOrInt = None) -> deque:
    """
    Creates a deque given ``maxlen``.
    Args:
        maxlen (BoolOrInt): maximum length of the deque. Can be either a ``bool`` or an ``int``. If ``bool``, then it ``maxlen``
            corresponds to 1 if ``False`` or ``None`` if ``True``. If an ``int``, then corresponds to the value.
    """

    return deque(maxlen=1 if maxlen is False else (None if isinstance(maxlen, bool) else maxlen))


class TensorContainer(object):
    """
    Implements a container for storing tuples that serialize/deserialize as tensors.
    """

    _PREFIX = "tensor_{type_:s}"
    _SEP = "__"

    def __init__(self, **kwargs: torch.Tensor):
        """
        Internal initializer for :class:`BufferIterable`.
        """

        super().__init__()

        self._tuples: Dict[str, Tuple[torch.Tensor, ...]] = OrderedDict([])
        self._deques: Dict[str, Deque[torch.Tensor]] = OrderedDict([])

        for k, v in kwargs.items():
            if v is not None:
                self.make_tuple(k, v)

    def make_tuple(self, name: str, values: torch.Tensor = None):
        """
        Creates a tuple named ``name`` with values ``torch.Tensor``.
        Args:
            name: name of the tuple
            values: the values to cast as a tuple.
        """

        self._tuples[name] = tuple(values) if values is not None else tuple()

    def make_deque(self, name: str, values: torch.Tensor = None, maxlen: int = None):
        # TODO: Copy form make_tuple
        """
        See :meth:`~BufferIterable.make_tuple`.
        Args:
            maxlen: the maximum length of the deque.
        """

        self._deques[name] = deque_ = make_dequeue(maxlen)

        if values is not None:
            deque_.extend(values)

    def get_as_tensor(self, key: str) -> torch.Tensor:
        """
        Gets the iterable as a tensor.

        Args:
            key (str): name of object to get.
        """

        to_stack = self.__getitem__(key)

        if to_stack:
            return torch.stack(tuple(to_stack), dim=0)

        return torch.tensor([])

    def __getitem__(self, key: str) -> Iterable[torch.Tensor]:
        if key in self._tuples:
            return self._tuples[key]
        if key in self._deques:
            return self._deques[key]

        raise Exception(f"Could not find '{key}'!")

    def __setitem__(self, key: str, parameter: Iterable[torch.Tensor]) -> None:
        raise Exception("Not allowed!")

    def __delitem__(self, key: str) -> None:
        if key in self._tuples:
            del self._tuples[key]
        else:
            del self._deques[key]

    def __len__(self) -> int:
        return len(self._tuples) + len(self._deques)

    def __iter__(self) -> Iterator[str]:
        return itertools.chain(self._tuples.keys(), self._deques.keys())

    def __contains__(self, key: str) -> bool:
        return any(key in item for item in [self._tuples, self._deques])

    def values(self) -> Iterable[Iterable[torch.Tensor]]:
        return itertools.chain(self._tuples.values(), self._deques.values())

    def keys(self):
        return itertools.chain(self._tuples.keys(), self._deques.keys())

    def state_dict(self):
        state_dict = OrderedDict([])

        for key, value in self._tuples.items():
            state_dict[self._PREFIX.format(type_="tuple") + self._SEP + key] = self.get_as_tensor(key)

        for key, value in self._deques.items():
            v = self.get_as_tensor(key)
            state_dict[self._PREFIX.format(type_=f"deque_{value.maxlen}") + self._SEP + key] = v

        return state_dict

    def load_state_dict(self, state_dict):
        p = self._PREFIX

        # TODO: Fix s.t. deque can be found...
        keys = tuple(u for u in state_dict.keys() if (p.format(type_="tuple") in u) or (p.format(type_="deque") in u))

        for k in keys:
            v = state_dict.pop(k)
            type_, name = k.replace("tensor_", "").split(self._SEP, 1)

            if type_ == "tuple":
                self.make_tuple(name, v)
            elif type_.startswith("deque"):
                _, maxlen = type_.split("_")
                self.make_deque(name, v, maxlen=int(maxlen) if maxlen != "None" else None)
