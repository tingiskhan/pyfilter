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

    # TODO: This is somewhat adhoc, should perhaps be done outside...
    if x.dim() > indices.dim():
        indices = indices.unsqueeze(-1).expand_as(x)

    return x.gather(dim, indices)
