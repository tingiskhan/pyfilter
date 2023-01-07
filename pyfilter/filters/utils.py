import torch


#: TODO: Improve this to allow for arbitrary batching...
def batched_gather(x: torch.Tensor, indices: torch.IntTensor, dim: int):
    """
    Similar to the gather method of :class:`torch.Tensor`.

    Args:
        x (torch.Tensor): tensor to resample.
        indices (torch.Tensor): indices to choose.
        dim (int): dimension to choose.

    Returns:
        Resampled ``x``.
    """

    if indices.dim() == 1:
        return x[indices]
    elif indices.dim() == 2:
        if x.dim() > indices.dim():
            indices = indices.unsqueeze(-1).repeat_interleave(x.shape[-1], dim=-1)

        return x.gather(dim, indices)

    raise NotImplementedError("Currently do not support more batch dimensions than 1!")
