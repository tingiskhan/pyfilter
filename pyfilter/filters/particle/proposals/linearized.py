import torch
from pyro.distributions import Normal
from torch.autograd import grad
from stochproc.timeseries import AffineProcess
from .base import Proposal


class Linearized(Proposal):
    """
    Given a state space model with dynamics
        .. math::
            Y_t \\sim p_\\theta(y_t \\mid X_t), \n
            X_{t+1} = f_\\theta(X_t) + g_\\theta(X_t) W_{t+1},

    where :math:`p_\\theta` denotes an arbitrary density parameterized by :math:`\\theta` and :math:`X_t`, and which is
    continuous and (twice) differentiable w.r.t. :math:`X_t`. This proposal seeks to approximate the optimal proposal
    density :math:`p_\\theta(y_t \\mid x_t) \\cdot p_\\theta(x_t \\mid x_{t-1})` by linearizing it around
    :math:`f_\\theta(x_t)` and approximate it using a normal distribution.
    """

    def __init__(self, n_steps=1, alpha: float = 1e-4, use_second_order: bool = False):
        """
        Initializes the ``Linearized`` proposal

        Args:
            n_steps: The number of steps to take when performing gradient descent
            alpha: The step size to take when performing gradient descent. Only matters when ``use_second_order`` is
                ``False``, or when the Hessian is badly conditioned.
            use_second_order: Whether to use second order information when constructing the proposal distribution.
                Amounts to using the diagonal of the Hessian.
        """
        super().__init__()
        self._alpha = alpha

        assert n_steps > 0, "``n_steps`` must be >= 1"

        self._n_steps = n_steps
        self._use_second_order = use_second_order

        self._is1d = None

    def set_model(self, model):
        if not isinstance(model.hidden, AffineProcess):
            raise ValueError(f"Hidden must be of type {AffineProcess.__class__.__name__}!")

        super(Linearized, self).set_model(model)
        self._is1d = self._model.hidden.n_dim == 0

        return self

    def sample_and_weight(self, y, x):
        mean, std = self._model.hidden.mean_scale(x)

        std = std.clone()
        x_copy = x.copy(values=mean)

        # TODO: Would optimally build density utilizing the mean and scale from above
        x_dist = self._model.hidden.build_density(x)

        for _ in range(self._n_steps):
            mean.requires_grad_(True)

            y_dist = self._model.build_density(x_copy)
            logl = y_dist.log_prob(y) + x_dist.log_prob(mean)
            g = grad(logl, mean, grad_outputs=torch.ones_like(logl), create_graph=self._use_second_order)[-1]

            ones_like_g = torch.ones_like(g)
            step = self._alpha * ones_like_g
            if self._use_second_order:
                neg_inv_hess = -grad(g, mean, grad_outputs=ones_like_g)[-1].pow(-1.0)

                # TODO: There is a better approach in Dahlin, find it
                mask = neg_inv_hess > 0.0

                step.masked_scatter_(mask, neg_inv_hess)
                std.masked_scatter_(mask, neg_inv_hess.sqrt())

                g.detach_()

            mean.detach_()
            mean += step * g

        kernel = Normal(mean, std, validate_args=False)
        if not self._is1d:
            kernel = kernel.to_event(1)

        x_result = x_copy.copy(values=kernel.sample)

        return x_result, self._weight_with_kernel(y, x_dist, x_result, kernel)
