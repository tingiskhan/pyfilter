import torch


# TODO: Really should return the object instead, this is sub-optimal
class StateVariable(torch.Tensor):
    """
    Implements a custom state variable for easier usage when performing indexing in functions, while still retaining
    compatibility with pytorch functionality.
    """

    def __new__(cls, data, *args, **kwargs):
        res = torch.Tensor._make_subclass(cls, data, *args, **kwargs)
        res.requires_grad = data.requires_grad

        return res

    def __init__(self, data, *args, **kwargs):
        self._helper = data

        if self.requires_grad:
            self.register_hook(self._grad_writer)

    def _grad_writer(self, grad):
        self._helper.grad = grad[:]

    def __getitem__(self, item):
        return self._helper[..., item]

    def get_base(self):
        """
        Returns the underlying tensor
        :rtype: torch.Tensor
        """

        return self._helper

