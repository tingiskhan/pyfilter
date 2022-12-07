import itertools
import warnings
from collections import OrderedDict, abc as container_abcs
from collections import deque
from typing import Optional, Mapping, Any, Iterator, Iterable, Tuple, Union, Dict, Deque

import torch
from torch.nn import Module

BoolOrInt = Union[int, bool]


def make_dequeue(maxlen: BoolOrInt = None) -> deque:
    """
    Creates a deque given ``maxlen``.
    Args:
        maxlen: The maximum length of the deque. Can be either a ``bool`` or an ``int``. If ``bool``, then it ``maxlen``
            corresponds to 1 if ``False`` or ``None`` if ``True``. If an ``int``, then corresponds to the value.
    """

    return deque(maxlen=1 if maxlen is False else (None if isinstance(maxlen, bool) else maxlen))


# TODO: Utilize torch's own BufferDict when available
class BufferDict(Module):
    """
    Implements a naive version of the :class:`torch.nn.ParameterDict` containing buffers instead of parameters. Note
    that this class basically copies :class:`torch.nn.ParameterDict`.
    """

    def __init__(self, parameters: Optional[Mapping[str, torch.Tensor]] = None) -> None:
        super(BufferDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._buffers[key]

    def __setitem__(self, key: str, parameter: torch.Tensor) -> None:
        self.register_buffer(key, parameter)

    def __delitem__(self, key: str) -> None:
        del self._buffers[key]

    def __setattr__(self, key: Any, value: Any) -> None:
        if isinstance(value, torch.nn.Parameter):
            warnings.warn(f"Setting attributes on '{self.__class__.__name__}' is not supported.")
        super(BufferDict, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._buffers)

    def __iter__(self) -> Iterator[str]:
        return iter(self._buffers.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._buffers

    def clear(self) -> None:
        """
        See ``torch.nn.ParameterDict``.
        """
        self._buffers.clear()

    def pop(self, key: str) -> torch.Tensor:
        """
        See ``torch.nn.ParameterDict``.
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        """
        See ``torch.nn.ParameterDict``.
        """
        return self._buffers.keys()

    def items(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        See ``torch.nn.ParameterDict``.
        """
        return self._buffers.items()

    def values(self) -> Iterable[torch.Tensor]:
        """
        See ``torch.nn.ParameterDict``.
        """
        return self._buffers.values()

    def update(self, parameters: Mapping[str, torch.Tensor]) -> None:
        """
        See ``torch.nn.ParameterDict``.
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                f"'{self.__class__.__name__}'.update should be called with an "
                "iterable of key/value pairs, but got " + type(parameters).__name__
            )

        if isinstance(parameters, (OrderedDict, BufferDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError(
                        f"'{self.__class__.__name__}' update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        f"'{self.__class__.__name__}' update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]


class BufferIterable(BufferDict):
    """
    Implements a container for storing tuples that serialized/deserialize as tensors.
    """

    _PREFIX = "tensor_{type_:s}"
    _SEP = "__"

    def __init__(self, **kwargs: torch.Tensor):
        """
        Initializes the :class:`BufferIterable` class.
        """

        super().__init__()

        self._tuples: Dict[str, Tuple[torch.Tensor, ...]] = OrderedDict([])
        self._deques: Dict[str, Deque[torch.Tensor]] = OrderedDict([])

        self._register_state_dict_hook(self._dump_hook)
        self._register_load_state_dict_pre_hook(self._load_hook)

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

    def __setattr__(self, key: Any, value: Any) -> None:
        if isinstance(value, torch.nn.Parameter):
            warnings.warn(f"Setting attributes on '{self.__class__.__name__}' is not supported.")
        super(BufferDict, self).__setattr__(key, value)

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

    def _apply(self, fn):
        super()._apply(fn)

        for key, item in self._tuples.items():
            as_tensor = self.get_as_tensor(key)
            new_tensor = fn(as_tensor)

            self.make_tuple(key, new_tensor)

        for key, item in self._deques.items():
            as_tensor = self.get_as_tensor(key)
            new_tensor = fn(as_tensor)

            self.make_deque(key, new_tensor, maxlen=item.maxlen)

        return self

    @classmethod
    def _dump_hook(cls, self: "BufferIterable", state_dict, prefix, local_metadata):
        for key, value in self._tuples.items():
            state_dict[prefix + cls._PREFIX.format(type_="tuple") + self._SEP + key] = self.get_as_tensor(key)

        for key, value in self._deques.items():
            v = self.get_as_tensor(key)
            state_dict[prefix + cls._PREFIX.format(type_=f"deque_{value.maxlen}") + self._SEP + key] = v

        return

    def _load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        p = self._PREFIX

        # TODO: Fix s.t. deque can be found...
        keys = tuple(u for u in state_dict.keys() if (p.format(type_="tuple") in u) or (p.format(type_="deque") in u))

        for k in keys:
            v = state_dict.pop(k)
            type_, name = k.replace("tensor_", "").split(self._SEP, 1)

            if prefix:
                type_ = type_.replace(prefix, "")

            if type_ == "tuple":
                self.make_tuple(name, v)
            elif type_.startswith("deque"):
                _, maxlen = type_.split("_")
                self.make_deque(name, v, maxlen=int(maxlen) if maxlen != "None" else None)

        return