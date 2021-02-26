import torch
from torch.distributions import Normal, Independent, TransformedDistribution, AffineTransform
from torch.autograd import grad
from typing import Optional
from ....timeseries import AffineProcess
from .base import Proposal


class Linearized(Proposal):
    def __init__(self, n_steps=1, alpha: Optional[float] = 0.25):
        """
        Implements a linearized proposal using Normal distributions. Do note that this proposal should be used for
        models that are log-concave in the observation density.

        :param n_steps: The number of steps to take when approximating the mean of the proposal density
        :param alpha: Takes step proportional to `alpha` if not None, else uses second order information
        """
        super().__init__()
        self._alpha = alpha
        self._n_steps = n_steps

    def set_model(self, model):
        if not (isinstance(model.observable, AffineProcess) and isinstance(model.hidden, AffineProcess)):
            raise ValueError(f"Both observable and hidden must be of type {AffineProcess.__class__.__name__}!")

        self._model = model

        return self

    def sample_and_weight(self, y, x):
        hidden_dist: TransformedDistribution = self._model.hidden.define_density(x)
        affine_transform = next(trans for trans in hidden_dist.transforms if isinstance(trans, AffineTransform))

        h_loc, h_scale = affine_transform.loc[:], affine_transform.scale[:]
        state = self._model.hidden.propagate_state(h_loc, x)

        for _ in range(self._n_steps):
            h_loc.requires_grad_(True)

            logl = self._model.observable.log_prob(y, state) + hidden_dist.log_prob(h_loc)
            g = grad(logl, h_loc, grad_outputs=torch.ones_like(logl), create_graph=self._alpha is None)[-1]

            if self._alpha is None:
                step = -1 / grad(g, h_loc, grad_outputs=torch.ones_like(g))[-1]
                std = step.sqrt()
            else:
                std = h_scale.detach()
                step = self._alpha

            h_loc = h_loc.detach() + step * g.detach()
            state = state.copy(h_loc)

        mean = h_loc.detach()

        if self._model.hidden_ndim == 0:
            kernel = Normal(mean, std)
        else:
            kernel = Independent(Normal(mean, std), self._model.hidden_ndim)

        new_x = self._model.hidden.propagate_state(kernel.sample(), x)

        return new_x, self._weight_with_kernel(y, new_x, hidden_dist, kernel)
