import torch
from torch.distributions import TransformedDistribution, Distribution, AffineTransform


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
