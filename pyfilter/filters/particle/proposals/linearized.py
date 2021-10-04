import torch
from torch.distributions import Normal, Independent, AffineTransform
from torch.autograd import grad
from ....timeseries import AffineProcess
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
        self._n_steps = n_steps
        self._use_second_order = use_second_order

        self._is1d = None

    def set_model(self, model):
        if not isinstance(model.hidden, AffineProcess):
            raise ValueError(f"Hidden must be of type {AffineProcess.__class__.__name__}!")

        self._model = model
        self._is1d = self._model.hidden.n_dim == 0

        return self

    def sample_and_weight(self, y, x):
        new_x = self._model.hidden.propagate(x)
        affine_transform = next(trans for trans in new_x.dist.transforms if isinstance(trans, AffineTransform))

        mean, h_scale = affine_transform.loc[:], affine_transform.scale[:]
        new_x.values = mean

        for _ in range(self._n_steps):
            mean.requires_grad_(True)

            y_dist = self._model.observable.build_density(new_x)

            logl = y_dist.log_prob(y) + new_x.dist.log_prob(mean)
            g = grad(logl, mean, grad_outputs=torch.ones_like(logl), create_graph=self._use_second_order)[-1]

            step = self._alpha * torch.ones_like(g)
            std = h_scale.clone()

            if self._use_second_order:
                neg_inv_hess = -1.0 / grad(g, mean, grad_outputs=torch.ones_like(g))[-1]
                mask = neg_inv_hess > 0.0

                step[mask] = neg_inv_hess[mask]
                std[mask] = neg_inv_hess[mask].sqrt()

                g.detach_()

            mean = mean.detach() + step * g
            new_x = new_x.copy(new_x.dist, mean)

        if self._is1d:
            kernel = Normal(mean, std, validate_args=False)
        else:
            kernel = Independent(Normal(mean, std), self._model.hidden.n_dim, validate_args=False)

        new_x = new_x.copy(new_x.dist, kernel.sample())

        return new_x, self._weight_with_kernel(y, new_x, kernel)
