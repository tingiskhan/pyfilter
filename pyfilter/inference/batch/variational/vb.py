from ..base import BaseBatchAlgorithm
import torch
from torch.optim import Adadelta as Adam, Optimizer
from .approximation import StateMeanField, ParameterMeanField
from ....timeseries import StateSpaceModel, StochasticProcess, TimeseriesState
from ....filters import UKF
from typing import Type, Union, Optional, Any, Dict
from .state import VariationalState
from ....constants import EPS


class BatchedState(TimeseriesState):
    pass


class VariationalBayes(BaseBatchAlgorithm):
    def __init__(
        self,
        model: Union[StateSpaceModel, StochasticProcess],
        samples=4,
        optimizer: Type[Optimizer] = Adam,
        max_iter=30e3,
        optkwargs: Optional[Dict[str, Any]] = None,
        use_filter=True,
    ):
        """
        Implements Variational Bayes for stochastic processes implementing either `StateSpaceModel` or
        `StochasticProcess`.
        :param model: The model
        :param samples: The number of samples
        :param optimizer: The optimizer
        :param max_iter: The maximum number of iterations
        :param optkwargs: Any optimizer specific kwargs
        :param use_filter: Whether to initialize VB with using filtered estimates
        """

        super().__init__(max_iter)
        self._model = model
        self._ns = samples

        self._use_filter = use_filter

        self._opt_type = optimizer
        self._optimizer = None
        self.optkwargs = optkwargs or dict()

    def is_converged(self, old_loss, new_loss):
        return ((new_loss - old_loss) ** 2) ** 0.5 < EPS

    def _fit(self, y: torch.Tensor, logging_wrapper, **kwargs) -> VariationalState:
        state = self.initialize(y, **kwargs)

        try:
            logging_wrapper.set_num_iter(self._max_iter)
            while not state.converged and state.iterations < self._max_iter:
                old_loss = state.loss

                state = self._step(y, state)
                logging_wrapper.do_log(state.iterations, self, y)

                state.iterations += 1
                state.converged = self.is_converged(old_loss, state.loss)

        except Exception as e:
            logging_wrapper.close()
            raise e

        logging_wrapper.close()

        return state

    # TODO: Fix this one
    def sample_parameter_approximation(self, param_approximation: ParameterMeanField):
        params = param_approximation.sample((self._ns,))

        for p in self._model.parameters():
            p.detach_()

        self._model.parameters_from_array(params, constrained=False)

        return self

    def loss(self, y, state):
        # ===== Sample parameters ===== #
        self.sample_parameter_approximation(state.param_approx)
        entropy = state.param_approx.entropy()

        if isinstance(self._model, StateSpaceModel):
            transformed = state.state_approx.sample(self._ns)

            x_t = transformed[:, 1:]
            x_tm1 = transformed[:, :-1]

            if self._model.hidden_ndim < 1:
                x_t.squeeze_(-1)
                x_tm1.squeeze_(-1)

            init_dist = self._model.hidden.initial_sample(as_dist=True)

            state_t = BatchedState(torch.arange(1, x_t.shape[0]), x_t)
            state_tm1 = BatchedState(torch.arange(x_t.shape[0] - 1), x_tm1)

            logl = (self._model.log_prob(y, state_t) + self._model.hidden.log_prob(x_t, state_tm1)).sum(1)
            logl += init_dist.log_prob(x_tm1[..., :1]).squeeze(-1)

            entropy += state.state_approx.entropy()

        else:
            logl = self._model.log_prob(y[1:], y[:-1]).sum(1)

        return -(logl.mean(0) + self._model.eval_prior_log_prob(constrained=False).mean() + entropy)

    def _seed_init_path(self, y) -> [int, torch.Tensor]:
        filt = UKF(self._model.copy()).set_nparallel(self._ns)
        state = filt.initialize()
        result = filt.longfilter(y, bar=False, init_state=state)

        maxind = result.loglikelihood.argmax()

        to_cat = (state.get_mean().mean(axis=1).unsqueeze(0), result.filter_means.squeeze(-1))
        return maxind, torch.cat(to_cat, axis=0)[:, maxind]

    def initialize(self, y, param_approx: ParameterMeanField, state_approx: Optional[StateMeanField] = None):
        self._model.sample_params((self._ns, 1))

        param_approx.initialize(self._model.priors())
        opt_params = param_approx.get_parameters()

        if isinstance(self._model, StateSpaceModel):
            state_approx.initialize(y, self._model.hidden)

            if self._use_filter:
                maxind, means = self._seed_init_path(y)

                state_approx._mean.data[:] = means
                param_approx._mean.data[:] = self._model.parameters_to_array(constrained=False)[maxind]

            opt_params += state_approx.get_parameters()

        optimizer = self._opt_type(opt_params, **self.optkwargs)

        return VariationalState(False, float("inf"), 0, param_approx, optimizer, state_approx)

    def _step(self, y, state):
        state.optimizer.zero_grad()
        elbo = state.loss = self.loss(y, state)
        elbo.backward()
        state.optimizer.step()

        return state
