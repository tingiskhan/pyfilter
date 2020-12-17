import torch


def enforce_tensor(func):
    def wrapper(obj, y, *args, **kwargs):
        if not isinstance(y, torch.Tensor):
            raise ValueError("The observation must be of type Tensor!")

        return func(obj, y, *args, **kwargs)

    return wrapper


def _construct_empty(array: torch.Tensor) -> torch.Tensor:
    """
    Constructs an empty array based on the shape.
    :param array: The array to reshape after
    """

    temp = torch.arange(array.shape[-1], device=array.device)
    return temp * torch.ones_like(array, dtype=temp.dtype)
