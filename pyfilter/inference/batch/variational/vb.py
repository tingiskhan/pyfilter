import torch
from .approximation import BaseApproximation
from .state import VariationalResult
from ..base import OptimizationBasedAlgorithm
from ....timeseries import StateSpaceModel, NewState


class VariationalBayes(OptimizationBasedAlgorithm):
    """
    Implements `Variational Bayes`.
    """

    def __init__(
        self,
        model,
        parameter_approximation: BaseApproximation,
        n_samples=4,
        max_iter=30e3,
        state_approximation: BaseApproximation = None,
        **kwargs,
    ):
        """
        Initializes the ``VariationalBayes`` class.

        Args:
             model: See base.
             n_samples: The number of Monte Carlo samples to use when approximating the loss.
             max_iter: See base.
             parameter_approximation: The type of variational approximation for the parameters to use.
             state_approximation: The type of variational approximation for the states to use.
             kwargs: See base.
        """

        super().__init__(model, max_iter, **kwargs)
        self._n_samples = int(n_samples)
        self._time_indices: torch.Tensor = None

        self._is_ssm = isinstance(self._model, StateSpaceModel)
        self._num_steps = None

        if state_approximation is None and isinstance(self._model, StateSpaceModel):
            raise Exception(f"You must pass ``state_approx`` if model is of type ``{StateSpaceModel.__class__}``")

        self._param_approx = parameter_approximation
        self._state_approx = state_approximation

    def loss(self, y, state: VariationalResult):
        param_dist = state.sample_and_update_parameters(self._model, (self._n_samples,))
        entropy = param_dist.entropy()

        if self._is_ssm:
            state_dist = self._state_approx.get_approximation()
            transformed = state_dist.rsample((self._n_samples,))

            x_t = transformed[:, 1:]
            x_tm1 = transformed[:, :-1]

            if self._model.hidden.n_dim < 1:
                x_t.squeeze_(-1)
                x_tm1.squeeze_(-1)

            y_state = NewState(self._time_indices[1 :: self._num_steps], values=x_t[:, :: self._num_steps])
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
            log_likelihood.mean(0) + self._model.eval_prior_log_prob(constrained=False).squeeze().mean() + entropy
        )

    def initialize(self, y):
        self._model.sample_params(torch.Size([self._n_samples, 1]))

        param_shape = self._model.concat_parameters(constrained=True, flatten=True).shape[-1:]
        self._param_approx.initialize(param_shape)

        opt_params = tuple(self._param_approx.parameters())

        t_end = y.shape[0]
        self._num_steps = self._model.hidden.num_steps if self._is_ssm else self._model.num_steps
        self._time_indices = torch.arange(0, t_end * self._num_steps + 1)

        if self._is_ssm:
            shape = torch.Size([self._time_indices.shape[-1], *self._model.hidden.initial_dist.event_shape])
            self._state_approx.initialize(shape)
            opt_params += tuple(self._state_approx.parameters())

        self.construct_optimizer(opt_params)

        return VariationalResult(False, torch.tensor(0.0), 0, self._param_approx, self._state_approx)
