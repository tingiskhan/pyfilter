import torch


def _construct_empty_index(array: torch.Tensor) -> torch.Tensor:
    """
    Constructs an empty array of indexes the same shape as ``array``.

    Args:
        array: The array to construct indexes for.
    """

    temp = torch.arange(array.shape[-1], device=array.device)
    return temp * torch.ones_like(array, dtype=temp.dtype)
