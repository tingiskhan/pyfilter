import torch
from typing import Optional
from torch.distributions import Distribution
from .approximation import StateMeanField, ParameterMeanField
from .state import VariationalState
from ..base import OptimizationBasedAlgorithm
from ...utils import params_from_tensor, eval_prior_log_prob, sample_model
from ....timeseries import StateSpaceModel, NewState


class VariationalBayes(OptimizationBasedAlgorithm):
    """
    Implements the `Variational Bayes` algorithm.
    """

    def __init__(self, model, n_samples=4, max_iter=30e3, **kwargs):
        """
        Initializes the ``VariationalBayes`` class.

        Args:
             model: See base.
             n_samples: The number of Monte Carlo samples to use when approximating the loss.
             max_iter: See base.
             kwargs: See base.
        """

        super().__init__(model, max_iter, **kwargs)
        self._n_samples = int(n_samples)
        self._time_indices: torch.Tensor = None

        self._is_ssm = isinstance(self._model, StateSpaceModel)
        self._num_steps = None

    def _construct_and_sample(self, param_approximation: ParameterMeanField) -> Distribution:
        param_dist = param_approximation.get_approximation()
        params = param_dist.rsample((self._n_samples,))

        for p in self._model.parameters():
            p.detach_()

        params_from_tensor(self._model, params, constrained=False)

        return param_dist

    def loss(self, y, state):
        param_dist = self._construct_and_sample(state.param_approx)
        entropy = param_dist.entropy()

        if self._is_ssm:
            state_dist = state.state_approx.get_approximation()
            transformed = state_dist.rsample((self._n_samples,))

            x_t = transformed[:, 1:]
            x_tm1 = transformed[:, :-1]

            if self._model.hidden.n_dim < 1:
                x_t.squeeze_(-1)
                x_tm1.squeeze_(-1)

            y_state = NewState(self._time_indices[1::self._num_steps], values=x_t[:, ::self._num_steps])
            x_state = NewState(self._time_indices[:-1], values=x_tm1)

            x_dist = self._model.hidden.build_density(x_state)
            y_dist = self._model.observable.build_density(y_state)

            log_likelihood = y_dist.log_prob(y).sum(1) + x_dist.log_prob(x_t).sum(1)
            log_likelihood += self._model.hidden.initial_dist.log_prob(x_tm1[:, :1]).squeeze()

            entropy += state_dist.entropy()

        else:
            state = NewState(self._time_indices, values=y[:-1])
            dist = self._model.build_density(state)

            log_likelihood = dist.log_prob(y[1:]).sum(1)
            log_likelihood += self._model.initial_dist.log_prob(y[0])

        return -(
            log_likelihood.mean(0) + eval_prior_log_prob(self._model, constrained=False).squeeze().mean() + entropy
        )

    def initialize(self, y, param_approx: ParameterMeanField, state_approx: Optional[StateMeanField] = None):
        if state_approx is None and isinstance(self._model, StateSpaceModel):
            raise Exception(f"You must pass ``state_approx`` if model is of type ``{StateSpaceModel.__class__}``")

        sample_model(self._model, torch.Size([self._n_samples, 1]))
        param_approx.initialize(y, self._model)

        opt_params = tuple(param_approx.parameters())

        t_end = y.shape[0]
        self._num_steps = self._model.hidden.num_steps if self._is_ssm else self._model.num_steps
        self._time_indices = torch.arange(0, t_end * self._num_steps)

        if self._is_ssm:
            state_approx.initialize(self._time_indices, self._model)
            opt_params += tuple(state_approx.parameters())

        optimizer = self._opt_type(opt_params, **self.opt_kwargs)

        return VariationalState(False, float("inf"), 0, param_approx, optimizer, state_approx)
