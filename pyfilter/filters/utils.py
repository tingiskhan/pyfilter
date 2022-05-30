import torch


#: TODO: Improve this to allow for arbitrary batching...
def batched_gather(x: torch.Tensor, indices: torch.IntTensor, dim: int):
    """
    Similar to the gather method of :class:`torch.Tensor`.

    Args:
        x: the tensor to select.
        indices: the indices to choose.
        dim: the dimension to choose.

    Returns:
        A selected tensor
    """

    if indices.dim() == 1:
        return x[indices]
    elif indices.dim() == 2:
        if x.dim() > indices.dim():
            indices = indices.unsqueeze(-1).repeat_interleave(2, dim=-1)

        return x.gather(dim, indices)

    raise NotImplementedError("Currently do not support more batch dimensions than 1!")
