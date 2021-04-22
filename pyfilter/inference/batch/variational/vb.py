import torch
from typing import Optional
from torch.distributions import Distribution
from .approximation import StateMeanField, ParameterMeanField
from .state import VariationalState
from ..base import OptimizationBasedAlgorithm
from ...utils import params_to_tensor, params_from_tensor, eval_prior_log_prob, sample_model
from ....timeseries import StateSpaceModel, NewState
from ....filters import UKF
from ....constants import EPS


class VariationalBayes(OptimizationBasedAlgorithm):
    """
    Implements Variational Bayes for stochastic processes implementing either `StateSpaceModel` or
    `StochasticProcess`.
    """

    def __init__(self, model, n_samples=4, max_iter=30e3, use_filter=True, **kwargs):
        super().__init__(model, max_iter, **kwargs)
        self._n_samples = n_samples
        self._use_filter = use_filter

    def is_converged(self, old_loss, new_loss):
        return (new_loss - old_loss).abs() < EPS

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

            state_t = NewState(torch.arange(1, x_t.shape[0]), values=x_t)
            state_tm1 = NewState(torch.arange(x_t.shape[0] - 1), values=x_tm1)

            x_dist = self._model.hidden.build_density(state_tm1)
            y_dist = self._model.observable.build_density(state_t)

            log_likelihood = (y_dist.log_prob(y) + x_dist.log_prob(x_t)).sum(1)
            log_likelihood += self._model.hidden.initial_dist.log_prob(x_tm1[:, :1]).squeeze()

            entropy += state_dist.entropy()

        else:
            state = NewState(torch.arange(1, y.shape[0]), values=y[:-1])
            dist = self._model.build_density(state)

            log_likelihood = dist.log_prob(y[1:]).sum(1)
            log_likelihood += self._model.initial_dist.log_prob(y[0])

        return -(log_likelihood.mean(0) + eval_prior_log_prob(self._model, constrained=False).mean() + entropy)

    def _seed_init_path(self, y) -> [int, torch.Tensor]:
        filt = UKF(self._model.copy()).set_nparallel(self._n_samples)
        state = filt.initialize()
        result = filt.longfilter(y, bar=False, init_state=state)

        maxind = result.loglikelihood.argmax()

        to_cat = (state.get_mean().mean(axis=1).unsqueeze(0), result.filter_means.squeeze(-1))
        return maxind, torch.cat(to_cat, axis=0)[:, maxind]

    def initialize(self, y, param_approx: ParameterMeanField, state_approx: Optional[StateMeanField] = None):
        is_ssm = isinstance(self._model, StateSpaceModel)

        sample_model(self._model, (self._n_samples, 1))
        param_approx.initialize(y, self._model)

        opt_params = tuple(param_approx.parameters())

        if is_ssm:
            state_approx.initialize(y, self._model)

            if self._use_filter:
                maxind, means = self._seed_init_path(y)

                state_approx.mean.data[:] = means
                param_approx.mean.data[:] = params_to_tensor(self._model, constrained=False)[maxind]

            opt_params += tuple(state_approx.parameters())

        optimizer = self._opt_type(opt_params, **self.opt_kwargs)

        return VariationalState(False, float("inf"), 0, param_approx, optimizer, state_approx)
