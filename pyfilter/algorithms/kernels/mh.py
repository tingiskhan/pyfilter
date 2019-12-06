from ...filters.base import BaseFilter
from ...utils import normalize, EPS
from .base import BaseKernel
from pyfilter.algorithms.utils import stacker, _eval_kernel, _construct_mvn, _mcmc_move
import torch
from torch.distributions import MultivariateNormal, Independent
import gpytorch
from gpytorch import kernels as k


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
        self._entire_hist = True

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

    def _calc_diff_logl(self, t_filt, filter_):
        """
        Helper method for calculating the acceptance probability.
        :param t_filt: The new filter
        :type t_filt: BaseFilter
        :param filter_: The old filter
        :type filter_: BaseFilter
        :return: Difference in loglikelihood
        :rtype: torch.Tensor
        """

        t_filt.reset().initialize().longfilter(self._y, bar=False)
        return t_filt.loglikelihood - filter_.loglikelihood

    def _before_resampling(self, filter_, stacked):
        """
        Helper method for carrying out operations before resampling.
        :param filter_: The filter
        :type filter_: BaseFilter
        :param stacked: The stacked parameters
        :type stacked: torch.Tensor
        :return: Self
        :rtype: ParticleMetropolisHastings
        """
        return self

    def _update(self, parameters, filter_, weights):
        for i in range(self._nsteps):
            # ===== Construct distribution ===== #
            stacked, mask = stacker(parameters, lambda u: u.t_values)
            dist = self.define_pdf(stacked, weights)

            # ===== Do stuff prior to saving ===== #
            self._before_resampling(filter_, stacked)

            # ===== Resample among parameters ===== #
            inds = self._resampler(weights, normalized=True)
            filter_.resample(inds, entire_history=self._entire_hist)

            # ===== Define new filters and move via MCMC ===== #
            t_filt = filter_.copy()
            t_filt.viewify_params((filter_._n_parallel, 1))
            _mcmc_move(t_filt.ssm.theta_dists, dist, mask, stacked.shape[0])

            # ===== Calculate difference in loglikelihood ===== #
            quotient = self._calc_diff_logl(t_filt, filter_)

            # ===== Calculate acceptance ratio ===== #
            plogquot = t_filt.ssm.p_prior() - filter_.ssm.p_prior()
            kernel = _eval_kernel(filter_.ssm.theta_dists, dist, t_filt.ssm.theta_dists)

            # ===== Check which to accept ===== #
            u = torch.empty_like(quotient).uniform_().log()
            toaccept = u < quotient + plogquot + kernel

            # ===== Update the description ===== #
            self.accepted = toaccept.sum().float() / float(toaccept.shape[0])

            if self._entire_hist:
                filter_.exchange(t_filt, toaccept)
            else:
                filter_.ssm.exchange(toaccept, t_filt.ssm)

            weights = normalize(filter_.loglikelihood)

        return True


class SymmetricMH(ParticleMetropolisHastings):
    def define_pdf(self, values, weights):
        return _construct_mvn(values, weights)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = k.ScaleKernel(k.MaternKernel(ard_num_dims=train_x.shape[-1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ApproximateParticleMetropolisHastings(ParticleMetropolisHastings):
    def __init__(self, block_len, training_iter=250, **kwargs):
        """
        Implements base class for approximate Particle MH.
        :param block_len: The size of the blocks to use
        :type block_len: int
        :param training_iter: The maximum amount of training iterations
        :type training_iter: int
        """

        super().__init__(**kwargs)
        self._bl = int(block_len)
        self._iters = int(training_iter)

        self._entire_hist = False
        self._model = None

    def _fit_gp(self, x, y):
        """
        Fits the Gaussian process.
        :param x: The parameters
        :param y: The likelihood
        :return:
        """
        # ===== Define model ===== #
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(x.device)
        model = ExactGPModel(x, y, likelihood).to(x.device) if self._model is None else self._model

        if self._model is not None:
            model.set_train_data(x, y)

        # ===== Set to train mode ===== #
        model.train()
        likelihood.train()

        # ===== Define optimizer ===== #
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # ===== Set loss ===== #
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # ===== Perform optimization ===== #
        old = torch.tensor(float('inf'))
        loss = -old
        i = 0
        while (loss - old).abs() > EPS and i <= self._iters:
            old = loss

            optimizer.zero_grad()

            output = model(x)
            loss = -mll(output, y)
            loss.backward()

            optimizer.step()

            i += 1

        return model

    def _calc_diff_logl(self, t_filt, filter_):
        stacked, _ = stacker(filter_.ssm.theta_dists, lambda u: u.t_values)
        n_stacked, _ = stacker(t_filt.ssm.theta_dists, lambda u: u.t_values)

        self._model.eval()

        return self._model(n_stacked).loc - self._model(stacked).loc

    def _before_resampling(self, filter_, stacked):
        self._model = self._fit_gp(stacked, sum(filter_.s_ll[-self._bl:]))

        return self


class ApproximateSymmetricMH(ApproximateParticleMetropolisHastings):
    def define_pdf(self, values, weights):
        return _construct_mvn(values, weights)