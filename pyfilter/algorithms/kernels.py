from ..filters.base import BaseFilter
from ..utils import EPS, normalize, add_dimensions
from ..timeseries.parameter import Parameter
import torch
from scipy.stats import chi2
from math import sqrt
from torch.distributions import MultivariateNormal, Independent


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


class BaseKernel(object):
    def __init__(self, record_stats=False, length=None):
        """
        The base kernel used for propagating parameters.
        :param record_stats: Whether to record the statistics
        :type record_stats: bool
        :param length: How many to record. If `None` records everything
        :type length: int
        """
        self._record_stats = record_stats

        self._recorded_stats = tuple()
        self._length = length
        self._resampler = None

    def _update(self, parameters, filter_, weights, *args):
        """
        Defines the function for updating the parameters for the user to override.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param filter_: The filter
        :type filter_: BaseFilter
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        raise NotImplementedError()

    def set_resampler(self, resampler):
        """
        Sets the resampler to use if necessary for kernel.
        :param resampler: The resampler
        :type resampler: callable
        :rtype: BaseKernel
        """
        self._resampler = resampler

        return self

    def update(self, parameters, filter_, weights, *args):
        """
        Defines the function for updating the parameters.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param filter_: The filter
        :type filter_: BaseFilter
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        w = normalize(weights)

        if self._record_stats:
            self.record_stats(parameters, w)

        self._update(parameters, filter_, w, *args)

        return self

    def record_stats(self, parameters, weights):
        """
        Records the stats of the parameters.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        res = tuple()
        for p in parameters:
            weights = add_dimensions(weights, p.dim())
            vals = p.t_values

            mean = (vals * weights).sum(0)
            scale = ((vals - mean) ** 2 * weights).sum(0).sqrt()

            res += ({
                'mean': mean,
                'scale': scale
            },)

        self._recorded_stats += (
            res,
        )

        if self._length is not None:
            self._recorded_stats = self._recorded_stats[-self._length:]

        return self

    def get_as_numpy(self):
        """
        Returns the stats numpy arrays instead of torch tensor.
        :rtype: tuple[tuple[dict[str, numpy.array]]
        """
        res = tuple()
        for pt in self._recorded_stats:
            temp = tuple()
            for p in pt:
                tempdict = dict()
                for k, v in p.items():
                    tempdict[k] = v.numpy()

                temp += (tempdict,)
            res += (temp,)

        return res


def _normal_test(x, alpha=0.05):
    """
    Implements a basic Jarque-Bera test for normality.
    :param x: The data
    :type x: torch.Tensor
    :param alpha: The level of confidence
    :type alpha: float
    :return: Whether a normal distribution or not
    :rtype: bool
    """
    mean = x.mean(0)
    var = ((x - mean) ** 2).mean(0)

    # ===== Skew ===== #
    skew = ((x - mean) ** 3).mean(0) / var ** 1.5

    # ===== Kurtosis ===== #
    kurt = ((x - mean) ** 4).mean(0) / var ** 2

    # ===== Statistic ===== #
    jb = x.shape[0] / 6 * (skew ** 2 + 1 / 4 * (kurt - 3) ** 2)

    return chi2(2).ppf(1 - alpha) >= jb


def _jitter(values, scale):
    """
    Jitters the parameters.
    :param values: The values
    :type values: torch.Tensor
    :param scale: The scaling to use for the variance of the proposal
    :type scale: float
    :return: Proposed values
    :rtype: torch.Tensor
    """

    return values + scale * torch.empty_like(values).normal_()


def _shrink(values, w, ess):
    """
    Shrinks the parameters towards their mean.
    :param values: The values
    :type values: torch.Tensor
    :param w: The normalized weights
    :type w: torch.Tensor
    :param ess: The previous ESS
    :type ess: float
    :return: The mean of the shrunk distribution and bandwidth
    :rtype: torch.Tensor, torch.Tensor
    """
    # ===== Calculate mean ===== #
    w = add_dimensions(w, values.dim())

    mean = (w * values).sum(0)

    # ===== Calculate STD ===== #
    norm_test = _normal_test(values)

    std = torch.empty_like(mean)
    var = torch.empty_like(mean)

    # ===== For those not normally distributed ===== #
    t_mask = ~norm_test
    if t_mask.any():
        sort, _ = values[:, t_mask].sort(0)
        std[t_mask] = (sort[int(0.75 * values.shape[0])] - sort[int(0.25 * values.shape[0])]) / 1.349

        var[t_mask] = std[t_mask] ** 2

    # ===== Those normally distributed ===== #
    if norm_test.any():
        var[norm_test] = (w * (values[:, norm_test] - mean[norm_test]) ** 2).sum(0)
        std[norm_test] = var[norm_test].sqrt()

    # ===== Calculate bandwidth ===== #
    bw = 1.59 * std * ess ** (-1 / 3)

    # ===== Calculate shrinkage and shrink ===== #
    beta = ((var - bw ** 2) / var).sqrt()

    return mean + beta * (values - mean), bw


def _unflattify(values, shape):
    """
    Unflattifies parameter values.
    :param values: The flattened array of values that are to be unflattified
    :type values: torch.Tensor
    :param shape: The shape of the parameter prior
    :type shape: torch.Size
    :rtype: torch.Tensor
    """

    if len(shape) < 1 or values.shape[1:] == shape:
        return values

    return values.reshape(values.shape[0], *shape)


class RegularJittering(BaseKernel):
    def __init__(self, p=4, **kwargs):
        """
        The regular jittering kernel proposed in the original paper.
        :param p: The p controlling the variance
        :type p: float
        """
        super().__init__(**kwargs)
        self._p = p

    def _update(self, parameters, filter_, weights, ess):
        # ===== Mutate parameters ===== #
        scale = 1 / sqrt(weights.numel() ** ((self._p + 2) / self._p))
        for p in parameters:
            p.t_values = _unflattify(_jitter(p.t_values, scale), p.c_shape)

        return self


class ShrinkageKernel(BaseKernel):
    """
    An improved regular shrinkage kernel, from the paper ..
    """

    def _update(self, parameters, filter_, weights, ess):
        # ===== Perform shrinkage ===== #
        stacked, mask = stacker(parameters, lambda u: u.t_values)
        shrunk_mean, scales = _shrink(stacked, weights, ess)

        # ===== Mutate parameters ===== #
        for msk, p in zip(mask, parameters):
            m, s = shrunk_mean[:, msk], scales[msk]

            p.t_values = _unflattify(_jitter(m, s), p.c_shape)

        return self


# TODO: The eps is completely arbitrary... but kinda influences the posterior
class AdaptiveShrinkageKernel(BaseKernel):
    def __init__(self, eps=1e-5, **kwargs):
        """
        Implements the adaptive shrinkage kernel of ..
        :param eps: The tolerance for when to stop shrinking
        :type eps: float
        """
        super().__init__(**kwargs)
        self._eps = eps
        self._old_var = None

    def _update(self, parameters, filter_, weights, ess):
        # ===== Perform shrinkage ===== #
        stacked, mask = stacker(parameters, lambda u: u.t_values)
        shrunk_mean, scales = _shrink(stacked, weights, ess)

        # ===== Check "convergence" ====== #
        w = add_dimensions(weights, stacked.dim())

        mean = (w * stacked).sum(0)
        var = (w * (stacked - mean) ** 2).sum(0)

        if self._old_var is None:
            var_diff = var
        else:
            var_diff = var - self._old_var

        # ===== Mutate parameters ===== #
        for p, msk in zip(parameters, mask):
            switched = (var_diff[msk].abs() < self._eps).all()
            m = (stacked if switched else shrunk_mean)[:, msk]

            p.t_values = _unflattify(_jitter(m, scales[msk]), p.c_shape)

        self._old_var = var

        return self
      

class EpachnikovKDE(BaseKernel):
    @staticmethod
    def _get_bandwidth(weights, values, ess):
        w = weights.unsqueeze(-1)
        mean = (values * w).sum(0)
        std = torch.empty_like(mean)

        norm_test = _normal_test(values)

        t_mask = ~norm_test
        if t_mask.any():
            sort, _ = values[:, t_mask].sort(0)
            std[t_mask] = (sort[int(0.75 * values.shape[0])] - sort[int(0.25 * values.shape[0])]) / 1.349
        if norm_test.any():
            std[norm_test] = (w * (values[:, norm_test] - mean[norm_test]) ** 2).sum(0).sqrt()

        # ===== Define "optimal" bw ===== #
        n = std.shape[-1]
        h = (ess * (n + 2) / 4) ** (-1 / (n + 4))

        return h * std

    def _generate_samples(self, values):
        """
        Samples from the epachnikov kernel.
        :rtype: torch.Tensor
        """
        ndim = values.shape[-1]
        is_samples = 5 * values.shape[0]    # TODO: Fix this

        hypercube = torch.empty((is_samples, ndim), device=values.device).uniform_(-1, 1)  # type: torch.Tensor
        norm = hypercube.norm(dim=-1)
        w = (1 - norm ** 2) * (norm < 1).float()

        normalized = normalize(w.log())
        inds = torch.multinomial(normalized, num_samples=values.shape[0], replacement=True)

        return hypercube[inds]

    def _update(self, parameters, filter_, weights, ess):
        values, mask = stacker(parameters, lambda u: u.t_values)

        # ===== Calculate covariance ===== #
        scale = self._get_bandwidth(weights, values, ess)

        # ===== Resample ===== #
        inds = self._resampler(weights, normalized=True)
        filter_.resample(inds, entire_history=False)

        # ===== Sample params ===== #
        samples = self._generate_samples(values)
        for p, msk in zip(parameters, mask):
            p.t_values = _unflattify(p.t_values + scale[msk] * samples[:, msk], p.c_shape)

        return self


class GaussianKDE(EpachnikovKDE):
    def _generate_samples(self, values):
        return torch.empty_like(values).normal_()


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
        p.t_values = _unflattify(rvs[:, msk], p.c_shape)

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


class ParticleMetropolisHastings(BaseKernel):
    def __init__(self, nsteps=1, **kwargs):
        """
        Implements a base class for the particle Metropolis Hastings class.
        :param nsteps: The number of steps to perform
        :type nsteps: int
        """
        super().__init__(**kwargs)

        self._nsteps = nsteps
        self._y = None
        self.accepted = None

    def set_data(self, y):
        """
        Sets the data to be used when calculating acceptance probabilities.
        :param y: The data
        :type y: tuple[torch.Tensor]
        :return: Self
        :rtype: ParticleMetropolisHastings
        """
        self._y = y

        return self

    def define_pdf(self, values, weights):
        """
        The method to be overridden by the user for defining the kernel to propagate the parameters. Note that the
        parameters are propagated in the transformed space.
        :param values: The parameters as a single Tensor
        :type values: torch.Tensor
        :param weights: The normalized weights of the particles
        :type weights: torch.Tensor
        :return: A distribution
        :rtype: MultivariateNormal|Independent
        """

        raise NotImplementedError()

    def _update(self, parameters, filter_, weights):
        for i in range(self._nsteps):
            # ===== Construct distribution ===== #
            stacked, mask = stacker(parameters, lambda u: u.t_values)
            dist = self.define_pdf(stacked, weights)

            # ===== Resample among parameters ===== #
            inds = self._resampler(weights, normalized=True)
            filter_.resample(inds, entire_history=True)

            # ===== Define new filters and move via MCMC ===== #
            t_filt = filter_.copy()
            t_filt.viewify_params((filter_._n_parallel, 1))
            _mcmc_move(t_filt.ssm.theta_dists, dist, mask, stacked.shape[0])

            # ===== Filter data ===== #
            t_filt.reset().initialize().longfilter(self._y, bar=False)

            quotient = t_filt.loglikelihood - filter_.loglikelihood

            # ===== Calculate acceptance ratio ===== #
            plogquot = t_filt.ssm.p_prior() - filter_.ssm.p_prior()
            kernel = _eval_kernel(filter_.ssm.theta_dists, dist, t_filt.ssm.theta_dists)

            # ===== Check which to accept ===== #
            u = torch.empty_like(quotient).uniform_().log()
            toaccept = u < quotient + plogquot + kernel

            # ===== Update the description ===== #
            self.accepted = toaccept.sum().float() / float(toaccept.shape[0])
            filter_.exchange(t_filt, toaccept)

            weights = normalize(filter_.loglikelihood)

        return self


class SymmetricMH(ParticleMetropolisHastings):
    def define_pdf(self, values, weights):
        mean = (values * weights.unsqueeze(-1)).sum(0)
        centralized = values - mean
        cov = torch.matmul(weights * centralized.t(), centralized)

        return MultivariateNormal(mean, scale_tril=torch.cholesky(cov))
