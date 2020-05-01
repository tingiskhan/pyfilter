from .base import BatchAlgorithm
import torch
from torch.optim import Adadelta as Adam
import tqdm
from .varapprox import StateMeanField, BaseApproximation, ParameterMeanField
from ..timeseries import StateSpaceModel
from ..timeseries.base import StochasticProcess
from .utils import stacker
from ..utils import EPS, unflattify
from ..filters import UKF


# TODO: Shape not working correctly when transformed != untransformed
class VariationalBayes(BatchAlgorithm):
    def __init__(self, model, samples=4, approx=None, optimizer=Adam, maxiter=30e3, optkwargs=None, use_filter=True):
        """
        Implements Variational Bayes for stochastic processes implementing either `StateSpaceModel` or
        `StochasticProcess`.
        :param model: The model
        :type model: StateSpaceModel|StochasticProcess
        :param samples: The number of samples
        :type samples: int
        :param approx: The variational approximation to use for the latent space
        :type approx: BaseApproximation
        :param optimizer: The optimizer
        :type optimizer: optim.Optimizer
        :param maxiter: The maximum number of iterations
        :type maxiter: int|float
        :param optkwargs: Any optimizer specific kwargs
        :type optkwargs: dict
        """

        super().__init__()
        self._model = model
        self._ns = samples

        # ===== Approximations ===== #
        self._is_ssm = isinstance(model, StateSpaceModel)
        self._s_approx = None

        if self._is_ssm:
            self._s_approx = approx or StateMeanField(model.hidden)

        self._p_approx = ParameterMeanField()

        # ===== Helpers ===== #
        self._mask = None
        self._runavg = 0.
        self._decay = 0.975
        self._use_filter = use_filter

        # ===== Optimization stuff ===== #
        self._opt_type = optimizer
        self._maxiters = int(maxiter)
        self.optkwargs = optkwargs or dict()

    @property
    def s_approximation(self):
        """
        Returns the resulting variational approximation of the states.
        :rtype: BaseApproximation
        """

        return self._s_approx

    @property
    def p_approximation(self):
        """
        Returns the resulting variational approximation of the parameters.
        :rtype: BaseApproximation
        """

        return self._p_approx

    def sample_params(self):
        """
        Samples parameters from the variational approximation.
        :return: Self
        :rtype: VariationalBayes
        """

        params = self._p_approx.sample(self._ns)
        for p, msk in zip(self._model.theta_dists, self._mask):
            p.detach_()
            p[:] = unflattify(p.bijection(params[:, msk]), p.c_shape)

        self._model.viewify_params((self._ns, 1))

        return self

    def _initialize(self, y):
        # ===== Sample model in place for a primitive version of initialization ===== #
        self._model.sample_params(self._ns)
        self._mask = stacker(self._model.theta_dists).mask    # NB: We create a mask once

        # ===== Setup the parameter approximation ===== #
        self._p_approx = self._p_approx.initialize(self._model.theta_dists)
        params = self._p_approx.get_parameters()

        # ===== Initialize the state approximation ===== #
        if self._is_ssm:
            self._s_approx.initialize(y)

            # ===== Run filter and use means for initialization ====== #
            if self._use_filter:
                filt = UKF(self._model.copy()).viewify_params((self._ns, 1)).set_nparallel(self._ns)
                filt.initialize().longfilter(y, bar=False)

                maxind = filt.result.loglikelihood.sum(0).argmax()
                self._s_approx._mean.data[1:] = filt.result.filter_means[:, maxind]

            # ===== Append parameters ===== #
            params += self._s_approx.get_parameters()

        return params

    def loss(self, y):
        """
        The loss function, i.e. ELBO.
        :rtype: torch.Tensor
        """
        # ===== Sample parameters ===== #
        self.sample_params()
        entropy = self._p_approx.entropy()

        if self._is_ssm:
            # ===== Sample states ===== #
            transformed = self._s_approx.sample(self._ns)

            # ===== Helpers ===== #
            x_t = transformed[:, 1:]
            x_tm1 = transformed[:, :-1]

            if self._model.hidden_ndim < 1:
                x_t.squeeze_(-1)
                x_tm1.squeeze_(-1)

            logl = self._model.log_prob(y, x_t) + self._model.h_weight(x_t, x_tm1)
            entropy += self._s_approx.entropy()

        else:
            logl = self._model.log_prob(y[1:], y[:-1])

        return -(logl.sum(1).mean(0) + torch.mean(self._model.p_prior(transformed=True), dtype=logl.dtype) + entropy)

    def _fit(self, y):
        optparams = self._initialize(y)

        elbo_old = -torch.tensor(float('inf'))
        elbo = -elbo_old

        optimizer = self._opt_type(optparams, **self.optkwargs)

        it = 0
        bar = tqdm.tqdm(total=self._maxiters)
        while (elbo - elbo_old).abs() > EPS and it < self._maxiters:
            elbo_old = elbo

            # ===== Perform optimization ===== #
            optimizer.zero_grad()
            elbo = self.loss(y)
            elbo.backward()
            optimizer.step()

            bar.update(1)

            if it > 0:
                self._runavg = self._runavg * self._decay - elbo * (1 - self._decay)
            else:
                self._runavg = -elbo

            bar.set_description(f'{str(self)} - Avg. ELBO: {self._runavg:.2f}')
            it += 1

        return self
