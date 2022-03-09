import torch
from torch.nn import Module
from typing import Optional, Mapping, Any, Iterator, Iterable, Tuple, Union, Dict
import warnings
from collections import OrderedDict, abc as container_abcs
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

    _PREFIX = "tensor_tuple__"

    def __init__(self, **kwargs: Iterable[torch.Tensor]):
        """
        Initializes the ``BufferIterable`` class.
        """

        super().__init__()
        self._iterables: Dict[str, Iterable[torch.Tensor]] = OrderedDict([])

        self._register_state_dict_hook(self._dump_hook)
        self._register_load_state_dict_pre_hook(self._load_hook)

        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def get_as_tensor(self, key: str) -> torch.Tensor:
        to_stack = self.__getitem__(key)

        if to_stack:
            return torch.stack(tuple(to_stack), dim=0)

        return torch.tensor([])

    def __getitem__(self, key: str) -> Iterable[torch.Tensor]:
        return self._iterables[key]

    def __setitem__(self, key: str, parameter: Iterable["Tensor"]) -> None:
        if isinstance(parameter, torch.Tensor):
            raise NotImplementedError(f"Currently does not support '{parameter.__class__.__name__}'")
        elif not isinstance(parameter, Iterable):
            raise ValueError(f"Must be of type {Iterable.__name__}!")

        self._iterables[key] = parameter

    def __delitem__(self, key: str) -> None:
        del self._iterables[key]

    def __setattr__(self, key: Any, value: Any) -> None:
        if isinstance(value, torch.nn.Parameter):
            warnings.warn(f"Setting attributes on '{self.__class__.__name__}' is not supported.")
        super(BufferDict, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._iterables)

    def __iter__(self) -> Iterator[str]:
        return iter(self._iterables.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._iterables

    def values(self) -> Iterable[Iterable["Tensor"]]:
        return self._iterables.values()

    def keys(self):
        return self._iterables.keys()

    def items(self) -> Iterable[Tuple[str, Iterable["Tensor"]]]:
        return self._iterables.items()

    def _apply(self, fn):
        super()._apply(fn)

        for key, item in self.items():
            as_tensor = self.get_as_tensor(key)
            new_tensor = fn(as_tensor)

            if isinstance(item, list):
                iterable = list(new_tensor)
            elif isinstance(item, tuple):
                iterable = tuple(new_tensor)
            elif isinstance(item, deque):
                iterable = deque(new_tensor, maxlen=item.maxlen)
            else:
                raise NotImplementedError(f"Does not support '{item.__class__.__name__}'")

            self.__setitem__(key, iterable)

        return self

    @classmethod
    def _dump_hook(cls, self, state_dict, prefix, local_metadata):
        for key, values in self._iterables.items():
            state_dict[prefix + cls._PREFIX + key] = self.get_as_tensor(key)

        return

    def _load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        p = prefix + self._PREFIX
        keys = [u for u in state_dict.keys() if u.startswith(p)]
        for k in keys:
            v = state_dict.pop(k)

            key = k.replace(p, "")
            item_to_add_to = self.__getitem__(key) if key in self else tuple()

            if isinstance(item_to_add_to, tuple):
                item_to_add_to += tuple(v)
            elif isinstance(item_to_add_to, list):
                item_to_add_to.extend(list(v))
            elif isinstance(item_to_add_to, deque):
                for item in v:
                    item_to_add_to.append(item)

            self.__setitem__(key, item_to_add_to)

        return
