from stochproc.timeseries import AffineProcess

from .base import Proposal
from .utils import find_mode_of_distribution


class Linearized(Proposal):
    r"""
    Given a state space model with dynamics
        .. math::
            Y_t \sim p_\theta(y_t \mid X_t), \newline
            X_{t+1} = f_\theta(X_t) + g_\theta(X_t) W_{t+1},

    where :math:`p_\theta` denotes an arbitrary density parameterized by :math:`\theta` and :math:`X_t`, and which is
    continuous and (twice) differentiable w.r.t. :math:`X_t`. This proposal seeks to approximate the optimal proposal
    density :math:`p_\theta(y_t \mid x_t) \cdot p_\theta(x_t \mid x_{t-1})` by linearizing it around
    :math:`f_\theta(x_t)` and approximate it using a normal distribution.
    """

    def __init__(self, n_steps=1, alpha: float = 1e-4, use_second_order: bool = False):
        """
        Internal initializer for :class:`Linearized`.

        Args:
            n_steps (int, optional): number of steps to take when performing gradient descent.. Defaults to 1.
            alpha (float, optional): step size to take when performing gradient descent. Only matters when ``use_second_order`` is
            ``False``, or when the Hessian is badly conditioned. Defaults to 1e-4.
            use_second_order (bool, optional): whether to use second order information when constructing the proposal distribution. Defaults to False.            
        """
        
        super().__init__()
        self._alpha = alpha

        assert n_steps > 0, "``n_steps`` must be >= 1"

        self._n_steps = n_steps
        self._use_second_order = use_second_order

    def set_model(self, model):
        if not isinstance(model.hidden, AffineProcess):
            raise ValueError(f"Hidden must be of type {AffineProcess.__class__.__name__}!")
        
        return super().set_model(model)
        
    def sample_and_weight(self, y, prediction):
        x = prediction.get_timeseries_state()
        
        # TODO: Re-use predictive density?
        x_dist = prediction.get_predictive_density(self._model)

        mean, std = self._model.hidden.mean_scale(x)
        initial_state = x.propagate_from(values=mean.clone())
        
        kernel = find_mode_of_distribution(self._model, x_dist, initial_state, std.clone(), y, self._n_steps, self._alpha, self._use_second_order)
        x_result = initial_state.copy(values=kernel.sample)

        return x_result, self._weight_with_kernel(y, x_dist, x_result, kernel)

    def copy(self) -> "Proposal":
        return Linearized(self._n_steps, self._alpha, self._use_second_order)
