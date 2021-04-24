import torch
from typing import Optional
from torch.distributions import Distribution
from .approximation import StateMeanField, ParameterMeanField
from .state import VariationalState
from ..base import OptimizationBasedAlgorithm
from ...utils import params_from_tensor, eval_prior_log_prob, sample_model
from ....timeseries import StateSpaceModel, NewState, StochasticDifferentialEquation
from ....constants import EPS


class VariationalBayes(OptimizationBasedAlgorithm):
    """
    Implements Variational Bayes for stochastic processes implementing either `StateSpaceModel` or
    `StochasticProcess`.
    """

    def __init__(self, model, n_samples=4, max_iter=30e3, **kwargs):
        super().__init__(model, max_iter, **kwargs)
        self._n_samples = int(n_samples)
        self._time_inds: torch.Tensor = None

        self._is_sde = isinstance(self._model.hidden, StochasticDifferentialEquation)
        self._step_size = self._model.hidden.num_steps if self._is_sde else 1

    def is_converged(self, old_loss, new_loss):
        return ((new_loss - old_loss).abs() < EPS) & (old_loss != new_loss)

    def sample_parameter_approximation(self, param_approximation: ParameterMeanField) -> Distribution:
        param_dist = param_approximation.dist()
        params = param_dist.rsample((self._n_samples,))

        for p in self._model.parameters():
            p.detach_()

        params_from_tensor(self._model, params, constrained=False)

        return param_dist

    def loss(self, y, state):
        param_dist = self.sample_parameter_approximation(state.param_approx)
        entropy = param_dist.entropy()

        if isinstance(self._model, StateSpaceModel):
            state_dist = state.state_approx.dist()
            transformed = state_dist.rsample((self._n_samples,))

            x_t = transformed[:, 1:]
            x_tm1 = transformed[:, :-1]

            if self._model.hidden.n_dim < 1:
                x_t.squeeze_(-1)
                x_tm1.squeeze_(-1)

            y_state = NewState(self._time_inds[1:], values=x_t)
            x_state = NewState(self._time_inds[:-1], values=x_tm1)

            x_dist = self._model.hidden.build_density(x_state)
            y_dist = self._model.observable.build_density(y_state)

            log_likelihood = y_dist.log_prob(y).sum(1) + x_dist.log_prob(x_t).sum(1)
            log_likelihood += self._model.hidden.initial_dist.log_prob(x_tm1[:, :1]).squeeze()

            entropy += state_dist.entropy()

        else:
            state = NewState(self._time_inds, values=y[:-1])
            dist = self._model.build_density(state)

            log_likelihood = dist.log_prob(y[1:]).sum(1)
            log_likelihood += self._model.initial_dist.log_prob(y[0])

        return -(log_likelihood.mean(0) + eval_prior_log_prob(self._model, constrained=False).squeeze().mean() + entropy)

    def initialize(self, y, param_approx: ParameterMeanField, state_approx: Optional[StateMeanField] = None):
        is_ssm = isinstance(self._model, StateSpaceModel)

        sample_model(self._model, (self._n_samples, 1))
        param_approx.initialize(y, self._model)

        opt_params = tuple(param_approx.parameters())
        self._time_inds = torch.arange(0, y.shape[0])

        if is_ssm:
            state_approx.initialize(self._time_inds, self._model)
            opt_params += tuple(state_approx.parameters())

        optimizer = self._opt_type(opt_params, **self.opt_kwargs)

        return VariationalState(False, float("inf"), 0, param_approx, optimizer, state_approx)
