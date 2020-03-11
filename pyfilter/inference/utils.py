import torch
from torch.distributions import MultivariateNormal
from ..utils import unflattify


class StackedObject(object):
    def __init__(self, concated, mask, prev_shape):
        """
        Helper object
        """

        self.concated = concated
        self.mask = mask
        self.prev_shape = prev_shape


def stacker(parameters, selector=lambda u: u.values, dim=1):
    """
    Stacks the parameters and returns a n-tuple containing the mask for each parameter.
    :param parameters: The parameters
    :type parameters: tuple[Parameter]|list[Parameter]
    :param selector: The selector
    :param dim: The dimension to start flattening from
    :type dim: int
    :rtype: StackedObject
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


def _mcmc_move(params, dist, stacked, shape):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params: The parameters to use for defining the distribution
    :type params: tuple[Parameter]
    :param dist: The distribution to use for sampling
    :type dist: MultivariateNormal
    :param stacked: The mask to apply for parameters
    :type stacked: StackedObject
    :param shape: The shape to sample
    :type shape: int
    :return: Samples from a multivariate normal distribution
    :rtype: torch.Tensor
    """

    rvs = dist.sample((shape,))

    for p, msk, ps in zip(params, stacked.mask, stacked.prev_shape):
        p.t_values = unflattify(rvs[:, msk], ps)

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

    p_vals = stacker(params, lambda u: u.t_values)
    n_p_vals = stacker(n_params, lambda u: u.t_values)

    return dist.log_prob(p_vals.concated) - dist.log_prob(n_p_vals.concated)