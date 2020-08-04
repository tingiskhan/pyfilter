import torch
from torch.distributions import MultivariateNormal, Distribution, Independent
from ..utils import unflattify
from typing import Iterable
from ..timeseries import Parameter
import warnings


class StackedObject(object):
    def __init__(self, concated: torch.Tensor, mask: torch.Tensor, prev_shape: torch.Size):
        """
        Object for storing the results of stacking tensors from `stacker`.
        """

        self.concated = concated
        self.mask = mask
        self.prev_shape = prev_shape


def stacker(parameters: Iterable[Parameter], selector=lambda u: u.values, dim=1):
    """
    Stacks the parameters and returns a n-tuple containing the mask for each parameter.
    :param parameters: The parameters
    :type parameters: tuple[Parameter]|list[Parameter]
    :param selector: The selector
    :param dim: The dimension to start flattening from
    """

    to_conc = tuple()
    mask = tuple()
    prev_shape = tuple()

    i = 0
    # TODO: Currently only supports one sampling dimension...
    for p in parameters:
        s = selector(p)
        flat = s if s.dim() <= dim else s.flatten(dim)

        if flat.dim() == dim:
            to_conc += (flat.unsqueeze(-1),)
            slc = i
        else:
            to_conc += (flat,)
            slc = slice(i, i + flat.shape[-1])

        mask += (slc,)
        i += to_conc[-1].shape[-1]
        prev_shape += (s.shape[dim:],)

    return StackedObject(torch.cat(to_conc, dim=-1), mask, prev_shape)


def _construct_mvn(x: torch.Tensor, w: torch.Tensor):
    """
    Constructs a multivariate normal distribution of weighted samples.
    :param x: The samples
    :param w: The weights
    """

    mean = (x * w.unsqueeze(-1)).sum(0)
    centralized = x - mean
    cov = torch.matmul(w * centralized.t(), centralized)

    return MultivariateNormal(mean, scale_tril=torch.cholesky(cov))


def _mcmc_move(params: Iterable[Parameter], dist: Distribution, stacked: StackedObject, shape: int):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params: The parameters to use for defining the distribution
    :param dist: The distribution to use for sampling
    :param stacked: The mask to apply for parameters
    :param shape: The shape to sample
    :return: Samples from a multivariate normal distribution
    """

    rvs = dist.sample(() if shape is None else (shape,))

    for p, msk, ps in zip(params, stacked.mask, stacked.prev_shape):
        p.t_values = unflattify(rvs[:, msk], ps)

    return True


def _eval_kernel(params: Iterable[Parameter], dist: Distribution, n_params: Iterable[Parameter]):
    """
    Evaluates the kernel used for performing the MCMC move.
    :param params: The current parameters
    :param dist: The distribution to use for evaluating the prior
    :param n_params: The new parameters to evaluate against
    :return: The log difference in priors
    """

    p_vals = stacker(params, lambda u: u.t_values)
    n_p_vals = stacker(n_params, lambda u: u.t_values)

    return dist.log_prob(p_vals.concated) - dist.log_prob(n_p_vals.concated)


def experimental(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn(f'{obj:s} is an experimental algorithm, use at own risk')

        return func(obj, *args, **kwargs)

    return wrapper


def preliminary(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn(f'{obj:s} is only a preliminary version algorithm, use at own risk')

        return func(obj, *args, **kwargs)

    return wrapper