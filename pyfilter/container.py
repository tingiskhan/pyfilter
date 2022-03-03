import torch
from torch.nn import Module
from typing import Optional, Mapping, Any, Iterator, Iterable, Tuple, Union, Dict
import warnings
from collections import OrderedDict, abc as container_abcs
from collections import deque

BoolOrInt = Union[int, bool]


def add_right(x: torch.Tensor, y: torch.Tensor, max_len: int = None):
    """
    Expands ``x`` to include ``y``.

    Args:
        x: The tensor to expand.
        y: The tensor to append.
        max_len: The maximum length of the tensor ``x``, acts as deque if specified.
    """

    with torch.no_grad():
        x.resize_((x.shape[0] + 1, *y.shape))
        x[-1] = y

        if max_len:
            x = x[-max_len:]

        return x


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
    Implements a naive version of the ``torch.nn.ParameterDict`` containing buffers instead of parameters. Note that
    this class basically copies ``torch.nn.ParameterDict``.
    """

    def __init__(self, parameters: Optional[Mapping[str, "Tensor"]] = None) -> None:
        super(BufferDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key: str) -> "Tensor":
        return self._buffers[key]

    def __setitem__(self, key: str, parameter: "Tensor") -> None:
        self.register_buffer(key, parameter)

    def __delitem__(self, key: str) -> None:
        del self._buffers[key]

    def __setattr__(self, key: Any, value: Any) -> None:
        if isinstance(value, torch.nn.Parameter):
            warnings.warn("Setting attributes on ParameterDict is not supported.")
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

    def pop(self, key: str) -> "Tensor":
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

    def items(self) -> Iterable[Tuple[str, "Tensor"]]:
        """
        See ``torch.nn.ParameterDict``.
        """
        return self._buffers.items()

    def values(self) -> Iterable["Tensor"]:
        """
        See ``torch.nn.ParameterDict``.
        """
        return self._buffers.values()

    def update(self, parameters: Mapping[str, "Tensor"]) -> None:
        """
        See ``torch.nn.ParameterDict``.
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParametersDict.update should be called with an "
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
                        "ParameterDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]


class BufferTuples(BufferDict):
    """
    Implements a container for storing tuples that serialized/deserialize as tensors.
    """

    _PREFIX = "tensor_tuple__"

    def __init__(self):
        """
        Initializes the ``BufferTuples`` class.
        """

        super().__init__()
        self._tuples: Dict[str, Tuple[torch.Tensor, ...]] = OrderedDict([])

        self._register_state_dict_hook(self._dump_hook)
        self._register_load_state_dict_pre_hook(self._load_hook)

    def __getitem__(self, key: str) -> Tuple[torch.Tensor, ...]:
        return self._tuples[key]

    def __setitem__(self, key: str, parameter: "Tensor") -> None:
        if isinstance(parameter, torch.Tensor):
            self._tuples[key] = (parameter,)
        elif isinstance(parameter, Iterable):
            self._tuples[key] = tuple(parameter)
        else:
            raise NotImplementedError(f"Currently does not support '{parameter.__class__.__name__}'")

    def __delitem__(self, key: str) -> None:
        del self._tuples[key]

    def __setattr__(self, key: Any, value: Any) -> None:
        if isinstance(value, torch.nn.Parameter):
            warnings.warn("Setting attributes on ParameterDict is not supported.")
        super(BufferDict, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._tuples)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tuples.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._tuples

    def values(self) -> Iterable[Tuple["Tensor", ...]]:
        return self._tuples.values()

    def keys(self):
        return self._tuples.keys()

    def items(self) -> Iterable[Tuple[str, Tuple["Tensor", ...]]]:
        return self._tuples.items()

    @classmethod
    def _dump_hook(cls, self, state_dict, prefix, local_metadata):
        for key, values in self._tuples.items():
            state_dict[prefix + cls._PREFIX + key] = torch.stack(values, dim=0)

        return

    def _load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        p = prefix + self._PREFIX
        keys = [u for u in state_dict.keys() if u.startswith(p)]
        for k in keys:
            v = state_dict.pop(k)
            self.__setitem__(k.replace(p, ""), tuple(v))

        return
