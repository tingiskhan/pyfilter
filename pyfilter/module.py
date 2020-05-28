import torch
from types import GeneratorType
from .timeseries.parameter import Parameter
from torch.distributions import Distribution, TransformedDistribution
from copy import deepcopy
from .utils import flatten
from typing import Union, Tuple, Type, Dict, Callable


_OBJTYPENAME = 'objtype'


class TensorContainerBase(object):
    @property
    def tensors(self) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError()

    # SEE: https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo=memo))

        return result


class TensorContainer(TensorContainerBase):
    def __init__(self, *args: torch.Tensor):
        if len(args) == 0:
            self._cont = tuple()
        else:
            self._cont = tuple(args) if not isinstance(args[0], GeneratorType) else tuple(*args)

    @property
    def tensors(self):
        return flatten(self._cont)

    def append(self, x: Union[torch.Tensor, TensorContainerBase, None]):
        if x is None:
            return

        if not isinstance(x, (torch.Tensor, TensorContainerBase)):
            raise NotImplementedError()

        self._cont += (x,)

        return self

    def extend(self, *x: torch.Tensor):
        if not (isinstance(x, tuple) and all(isinstance(t, torch.Tensor) for t in x)):
            raise NotImplementedError()

        self._cont += x

        return self

    def __getitem__(self, item):
        return self._cont[item]

    def __iter__(self):
        return (t for t in self._cont)

    def __bool__(self):
        return not not self._cont

    def __len__(self):
        return len(self._cont)


class TensorContainerDict(TensorContainerBase):
    def __init__(self, **kwargs):
        self._dict = dict(**kwargs)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, item):
        return self._dict[item]

    def __bool__(self):
        return not not self._dict

    @property
    def tensors(self):
        return flatten(self.values())

    def items(self):
        return self._dict.items()

    def values(self):
        if all(isinstance(v, TensorContainerBase) for v in self._dict.values()):
            return tuple(d.values() for d in self._dict.values())

        return tuple(self._dict.values())


def _find_types(x, type_: Type) -> Dict[str, object]:
    """
    Helper method for finding all type_ in x.
    :param x: The object
    :return: Dictionary
    """

    return {k: v for k, v in vars(x).items() if isinstance(v, type_)}


# TODO: Wait for pytorch to implement moving entire distributions
def _iterate_distribution(d: Distribution) -> Tuple[Distribution, ...]:
    """
    Helper method for iterating over distributions.
    :param d: The distribution
    """

    res = tuple()
    if not isinstance(d, TransformedDistribution):
        res += tuple(_find_types(d, torch.Tensor).values())

        for sd in _find_types(d, Distribution).values():
            res += _iterate_distribution(sd)

    else:
        res += _iterate_distribution(d.base_dist)

        for t in d.transforms:
            res += tuple(_find_types(t, torch.Tensor).values())

    return res


class Module(object):
    def _find_obj_helper(self, type_: Type):
        """
        Helper object for finding a specific type of objects in self.
        :param type_: The type to filter on
        """

        return _find_types(self, type_)

    def modules(self):
        """
        Finds and returns all instances of type module.
        """

        return self._find_obj_helper(Module)

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        """
        Finds and returns all instances of type module.
        """

        res = tuple()

        # ===== Find all tensor types ====== #
        res += tuple(self._find_obj_helper(torch.Tensor).values())

        # ===== Tensor containers ===== #
        for tc in self._find_obj_helper(TensorContainerBase).values():
            res += tc.tensors
            for t in (t_ for t_ in tc.tensors if isinstance(t_, Parameter) and t_.trainable):
                res += _iterate_distribution(t.distr)

        # ===== Pytorch distributions ===== #
        for d in self._find_obj_helper(Distribution).values():
            res += _iterate_distribution(d)

        # ===== Modules ===== #
        for mod in self.modules().values():
            res += mod.tensors()

        return res

    def apply(self, f: Callable[[torch.Tensor], torch.Tensor]):
        """
        Applies function f to all tensors.
        :param f: The callable
        :return: Self
        """

        for t in (t_ for t_ in self.tensors() if t_._base is None):
            t.data = f(t.data)

            if t._grad is not None:
                t._grad.data = f(t._grad.data)

        for t in (t_ for t_ in self.tensors() if t_._base is not None):
            # TODO: Not too sure about this one, happens for some distributions
            if t._base.dim() > 0:
                t.data = t._base.data.view(t.data.shape)
            else:
                t.data = f(t.data)

        return self

    def to_(self, device: str):
        """
        Move to device.
        :param device: The device to move to
        :return: Self
        """

        return self.apply(lambda u: u.to(device))

    def state_dict(self) -> Dict[str, object]:
        """
        Returns the state dictionary.
        """
        res = dict()
        res[_OBJTYPENAME] = self.__class__.__name__

        # ===== Tensors ===== #
        tens = self._find_obj_helper(torch.Tensor)
        res.update(tens)

        # ===== Tensor containers ===== #
        conts = self._find_obj_helper(TensorContainerBase)
        res.update(conts)

        # ===== Modules ===== #
        modules = self.modules()

        for k, m in modules.items():
            res[k] = m.state_dict()

        return res

    def load_state_dict(self, state: Dict[str, object]):
        """
        Loads the state dictionary.
        :param state: The state dictionary
        :return: Self
        """

        from .timeseries.base import StochasticProcess

        if state[_OBJTYPENAME] != self.__class__.__name__:
            raise ValueError(f'Cannot cast {state[_OBJTYPENAME]} as {self.__class__.__name__}!')

        for k, v in ((k_, v_) for k_, v_ in state.items() if k_ != _OBJTYPENAME):
            attr = getattr(self, k)

            if isinstance(attr, Module):
                attr.load_state_dict(v)

                if isinstance(attr, StochasticProcess):
                    attr.viewify_params(torch.Size([]))

            elif isinstance(v, TensorContainerBase) and all(isinstance(i, Parameter) for i in v.tensors):
                for new, old in zip(v.tensors, getattr(self, k).tensors):
                    new._prior = old._prior

                setattr(self, k, v)
            else:
                setattr(self, k, v)

        return self
