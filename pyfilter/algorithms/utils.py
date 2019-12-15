import torch
from torch.distributions import MultivariateNormal
from ..utils import unflattify


def stacker(parameters, selector=lambda u: u.values):
    """
    Stacks the parameters and returns a n-tuple containing the mask for each parameter.
    :param parameters: The parameters
    :type parameters: tuple[Parameter]|list[Parameter]
    :param selector: The selector
    :rtype: torch.Tensor, tuple[slice]
    """

    to_conc = tuple()
    mask = tuple()

    i = 0
    for p in parameters:
        if p.c_numel() < 2:
            to_conc += (selector(p).unsqueeze(-1),)
            slc = i
        else:
            to_conc += (selector(p).flatten(1),)
            slc = slice(i, i + p.c_numel())

        mask += (slc,)
        i += p.c_numel()

    return torch.cat(to_conc, dim=-1), mask


def _construct_mvn(x, w):
    """
    Constructs a multivariate normal distribution of weighted samples.
    :param x: The samples
    :type x: torch.Tensor
    :param w: The weights
    :type w: torch.Tensor
    :rtype: MultivariateNormal
    """

    mean = (x * w.unsqueeze(-1)).sum(0)
    centralized = x - mean
    cov = torch.matmul(w * centralized.t(), centralized)

    return MultivariateNormal(mean, scale_tril=torch.cholesky(cov))


def _mcmc_move(params, dist, mask, shape):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params: The parameters to use for defining the distribution
    :type params: tuple[Parameter]
    :param dist: The distribution to use for sampling
    :type dist: MultivariateNormal
    :param mask: The mask to apply for parameters
    :type mask: tuple[slice]
    :param shape: The shape to sample
    :type shape: int
    :return: Samples from a multivariate normal distribution
    :rtype: torch.Tensor
    """

    rvs = dist.sample((shape,))

    for p, msk in zip(params, mask):
        p.t_values = unflattify(rvs[:, msk], p.c_shape)

    return True


def _eval_kernel(params, dist, n_params):
    """
    Evaluates the kernel used for performing the MCMC move.
    :param params: The current parameters
    :type params: tuple[Distribution]
    :param dist: The distribution to use for evaluating the prior
    :type dist: MultivariateNormal
    :param n_params: The new parameters to evaluate against
    :type n_params: tuple of Distribution
    :return: The log difference in priors
    :rtype: torch.Tensor
    """

    p_vals, _ = stacker(params, lambda u: u.t_values)
    n_p_vals, _ = stacker(n_params, lambda u: u.t_values)

    return dist.log_prob(p_vals) - dist.log_prob(n_p_vals)