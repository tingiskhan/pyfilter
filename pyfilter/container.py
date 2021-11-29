import torch
from torch.nn import Module
from typing import Optional, Mapping, Any, Iterator, Iterable, Tuple, Union
import warnings
from collections import OrderedDict, abc as container_abcs
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from collections import deque

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


class TensorTuple(IterableDataset):
    """
    Implements a tuple like tensor storage.
    """

    def __init__(self, *tensors, maxlen=None):
        self.tensors = make_dequeue(maxlen=maxlen)
        self.tensors.extend(tensors)

    def __iter__(self) -> Iterator[T_co]:
        for t in self.tensors:
            yield t

    def __getitem__(self, index) -> T_co:
        return self.tensors[index]

    def __add__(self, other: IterableDataset):
        return TensorTuple(*self.tensors, *other.tensors, maxlen=max(self.tensors.maxlen, other.tensors.maxlen))

    def values(self) -> torch.Tensor:
        return torch.stack(tuple(self.tensors), dim=0)

    def append(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Can only concatenate tensors, not {tensor.__class__.__name__}!")

        self.tensors.append(tensor)

    def apply(self, fn):
        new_deque = make_dequeue(maxlen=self.tensors.maxlen)

        while self.tensors:
            for t in self.tensors.popleft():
                new_deque.append(fn(t))

        self.tensors = new_deque
