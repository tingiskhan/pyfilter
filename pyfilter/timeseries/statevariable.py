import torch
from ..utils import concater


# TODO: Don't know if good, but works
class StateVariable(torch.Tensor):
    """
    Implements a custom state variable for easier usage when performing indexing in functions, while still retaining
    compatibility with pytorch functionality.
    """

    def __new__(cls, data):
        out = torch.Tensor._make_subclass(cls, data)

        out.requires_grad_(data.requires_grad)
        out._tempdata = data

        return out

    @property
    def data(self):
        return self._tempdata

    def __getitem__(self, item):
        return self._tempdata[..., item]

    def __setitem__(self, key, value):
        self._tempdata[..., key] = value


def tensor_caster(func):
    """
    Function for helping out when it comes to multivariate models. Returns a torch.Tensor
    :param func: The function to pass
    :type func: callable
    :rtype: torch.Tensor
    """

    def wrapper(obj, x):
        if obj._inputdim > 1 and not isinstance(x, StateVariable):
            x = StateVariable(x)

        res = concater(func(obj, x))

        if obj.ndim > 1:
            return StateVariable(res)

        return res

    return wrapper