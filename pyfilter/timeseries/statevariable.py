import torch


# TODO: Don't know if good, but works
class StateVariable(torch.Tensor):
    """
    Implements a custom state variable for easier usage when performing indexing in functions, while still retaining
    compatibility with pytorch functionality.
    """

    def __new__(cls, data, *args, **kwargs):
        return torch.Tensor._make_subclass(cls, data, *args, **kwargs)

    def __init__(self, x, *args, **kwargs):
        self._tempdata = x

    def __getitem__(self, item):
        return self._tempdata[..., item]

    def __setitem__(self, key, value):
        self._tempdata[..., key] = value
        self.data[key] = value