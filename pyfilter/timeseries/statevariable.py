import torch


class StateVariable(torch.Tensor):
    """
    Implements a custom state variable for easier usage when performing indexing in functions, while still retaining
    compatibility with pytorch functionality.
    """

    def __getitem__(self, item):
        return self.data[..., item]

    def __setitem__(self, key, value):
        self.data[..., key] = value


def tensor_caster(func):
    """
    Function for helping out when it comes to multivariate models. Returns a torch.Tensor
    :param func: The function to pass
    :type func: callable
    :rtype: torch.Tensor
    """

    def wrapper(obj, x):
        if obj._inputdim > 1:
            x = StateVariable(x)

        out = func(obj, x)

        if not isinstance(out, tuple):
            return out

        return torch.cat(tuple(tx.unsqueeze(-1) for tx in out), dim=-1)

    return wrapper