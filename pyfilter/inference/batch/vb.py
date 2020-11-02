from .base import OptimizationBatchAlgorithm
import torch
from torch.optim import Adadelta as Adam, Optimizer
from .varapprox import StateMeanField, ParameterMeanField
from ...timeseries import StateSpaceModel, StochasticProcess
from ...utils import unflattify, stacker
from ...filters import APF
from typing import Type, Union, Optional, Any, Dict
from .state import VariationalState


class VariationalBayes(OptimizationBatchAlgorithm):
    def __init__(self, model: Union[StateSpaceModel, StochasticProcess], samples=4, optimizer: Type[Optimizer] = Adam,
                 max_iter=30e3, optkwargs: Optional[Dict[str, Any]] = None, use_filter=True):
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

        # ===== Helpers ===== #
        self._mask = None
        self._use_filter = use_filter

        # ===== Optimization stuff ===== #
        self._opt_type = optimizer
        self._optimizer = None
        self.optkwargs = optkwargs or dict()

    def sample_params(self, param_approximation: ParameterMeanField):
        """
        Samples parameters from the variational approximation.
        :return: Self
        """

        params = param_approximation.sample(self._ns)
        for p, msk in zip(self._model.parameter_distributions, self._mask):
            p.detach_()
            p[:] = unflattify(p.bijection(params[:, msk]), p.c_shape)

        self._model.viewify_params((self._ns, 1))

        return self

    def loss(self, y, state):
        """
        The loss function, i.e. ELBO.
        """
        # ===== Sample parameters ===== #
        self.sample_params(state.param_approx)
        entropy = state.param_approx.entropy()

        if isinstance(self._model, StateSpaceModel):
            # ===== Sample states ===== #
            transformed = state.state_approx.sample(self._ns)

            # ===== Helpers ===== #
            x_t = transformed[:, 1:]
            x_tm1 = transformed[:, :-1]

            if self._model.hidden_ndim < 1:
                x_t.squeeze_(-1)
                x_tm1.squeeze_(-1)

            init_dist = self._model.hidden.i_sample(as_dist=True)

            logl = (self._model.log_prob(y, x_t) + self._model.h_weight(x_t, x_tm1)).sum(1)
            logl += init_dist.log_prob(x_tm1[..., :1]).squeeze(-1)

            entropy += state.state_approx.entropy()

        else:
            logl = self._model.log_prob(y[1:], y[:-1]).sum(1)

        return -(logl.mean(0) + self._model.p_prior(transformed=True).mean() + entropy)

    def _seed_init_path(self, y) -> [int, torch.Tensor]:
        filt = APF(self._model.copy(), 1000).set_nparallel(self._ns).viewify_params((self._ns, 1))
        state = filt.initialize()
        result = filt.longfilter(y, bar=False, init_state=state)

        maxind = result.loglikelihood.argmax()

        return maxind, torch.cat((state.x.mean(axis=1).unsqueeze(0), result.filter_means), axis=0)[:, maxind]

    def initialize(self, y, param_approx: ParameterMeanField, state_approx: Optional[StateMeanField] = None):
        # ===== Sample model in place for a primitive version of initialization ===== #
        self._model.sample_params(self._ns)
        self._mask = stacker(self._model.parameter_distributions).mask  # NB: We create a mask once

        # ===== Setup the parameter approximation ===== #
        param_approx.initialize(self._model.parameter_distributions)
        opt_params = param_approx.get_parameters()

        # ===== Initialize the state approximation ===== #
        if isinstance(self._model, StateSpaceModel):
            state_approx.initialize(y, self._model.hidden)

            # ===== Run filter and use means for initialization ====== #
            if self._use_filter:
                maxind, means = self._seed_init_path(y)
                state_approx._mean.data[:] = means
                state_approx._log_std.data[:] = -1.

                param_approx._mean.data[:] = self._model.parameters_as_matrix().concated[maxind]

            # ===== Append parameters ===== #
            opt_params += state_approx.get_parameters()

        optimizer = self._opt_type(opt_params, **self.optkwargs)

        return VariationalState(False, float("inf"), 0, param_approx, optimizer, state_approx)

    def _step(self, y, state):
        state.optimizer.zero_grad()
        elbo = state.loss = self.loss(y, state)
        elbo.backward()
        state.optimizer.step()

        return state
