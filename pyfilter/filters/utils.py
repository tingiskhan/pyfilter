import torch
from torch.distributions import TransformedDistribution, Distribution, AffineTransform
import einops


def _construct_empty_index(array: torch.Tensor) -> torch.Tensor:
    """
    Constructs an empty array of indexes the same shape as ``array``.

    Args:
        array: The array to construct indexes for.
    """

    temp = torch.arange(array.shape[-1], device=array.device)
    return temp * torch.ones_like(array, dtype=temp.dtype)


def select_mean_of_dist(dist: Distribution) -> torch.Tensor:
    """
    Helper method for selecting the mean/location attribute of a given distribution.

    Args:
        dist: The distribution for which to get the mean/location for.
    """

    try:
        return dist.mean
    except NotImplementedError as e:
        if isinstance(dist, TransformedDistribution):
            if not isinstance(dist.transforms[0], AffineTransform):
                raise e

            return dist.transforms[0].loc
        else:
            raise Exception(f"Currently cannot handle '{dist.__class__.__name__}'!")
    except Exception as e:
        raise e


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
