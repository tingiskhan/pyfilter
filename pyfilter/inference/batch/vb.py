from .base import OptimizationBatchAlgorithm
import torch
from torch.optim import Adadelta as Adam, Optimizer
from .varapprox import StateMeanField, BaseApproximation, ParameterMeanField
from ...timeseries import StateSpaceModel, StochasticProcess
from ...utils import EPS, unflattify, stacker
from ...filters import UKF
from typing import Type, Union, Optional, Any, Dict
from .state import VariationalState


class VariationalBayes(OptimizationBatchAlgorithm):
    def __init__(self, model: Union[StateSpaceModel, StochasticProcess], samples=4, approx: BaseApproximation = None,
                 optimizer: Type[Optimizer] = Adam, max_iter=30e3, optkwargs: Optional[Dict[str, Any]] = None,
                 use_filter=True):
        """
        Implements Variational Bayes for stochastic processes implementing either `StateSpaceModel` or
        `StochasticProcess`.
        :param model: The model
        :param samples: The number of samples
        :param approx: The variational approximation to use for the latent space
        :param optimizer: The optimizer
        :param max_iter: The maximum number of iterations
        :param optkwargs: Any optimizer specific kwargs
        :param use_filter: Whether to initialize VB with using filtered estimates
        """

        super().__init__(max_iter)
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
        self._optimizer = None
        self.optkwargs = optkwargs or dict()

    def sample_params(self):
        """
        Samples parameters from the variational approximation.
        :return: Self
        """

        params = self._p_approx.sample(self._ns)
        for p, msk in zip(self._model.theta_dists, self._mask):
            if not p.trainable:
                continue

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
                filt = UKF(self._model.copy()).set_nparallel(self._ns)
                result = filt.longfilter(y, bar=False)

                maxind = result.loglikelihood.argmax()

                filtres = result.filter_means[:, maxind]
                if self._model.hidden_ndim < 1:
                    filtres.squeeze_(-1)

                self._s_approx._mean.data[1:] = filtres

            # ===== Append parameters ===== #
            params += self._s_approx.get_parameters()

        return params

    def loss(self, y: torch.Tensor):
        """
        The loss function, i.e. ELBO.
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

            init_dist = self._model.hidden.i_sample(as_dist=True)

            logl = (self._model.log_prob(y, x_t) + self._model.h_weight(x_t, x_tm1)).sum(1)
            logl += init_dist.log_prob(x_tm1[..., :1]).squeeze(-1)

            entropy += self._s_approx.entropy()

        else:
            logl = self._model.log_prob(y[1:], y[:-1]).sum(1)

        return -(logl.mean(0) + self._model.p_prior(transformed=True).mean() + entropy)

    def is_converged(self, old_loss, new_loss):
        return (new_loss - old_loss).abs() < EPS

    def initialize(self, y):
        self._optimizer = self._opt_type(self._initialize(y), **self.optkwargs)

        return self

    def _step(self, y):
        self._optimizer.zero_grad()
        elbo = self.loss(y)
        elbo.backward()
        self._optimizer.step()

        return elbo

    def _fit(self, y: torch.Tensor, logging_wrapper, **kwargs) -> VariationalState:
        state = super(VariationalBayes, self)._fit(y, logging_wrapper, **kwargs)

        return VariationalState(state.converged, state.final_loss, state.iterations, self._p_approx, self._s_approx)
