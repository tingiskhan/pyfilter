import torch
from torch.distributions import Distribution, TransformedDistribution
from typing import Tuple, Type, Dict, Callable


_OBJTYPENAME = 'objtype'


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
        raise NotImplementedError()

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

        res.update(**self.populate_state_dict())

        return res

    def populate_state_dict(self) -> Dict[str, object]:
        raise NotImplementedError()

    def load_state_dict(self, state: Dict[str, object]):
        """
        Loads the state dictionary.
        :param state: The state dictionary
        :return: Self
        """

        raise NotImplementedError()
