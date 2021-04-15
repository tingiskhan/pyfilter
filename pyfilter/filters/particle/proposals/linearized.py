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
        self._is1d = None

    def set_model(self, model):
        if not (isinstance(model.observable, AffineProcess) and isinstance(model.hidden, AffineProcess)):
            raise ValueError(f"Both observable and hidden must be of type {AffineProcess.__class__.__name__}!")

        self._model = model
        self._is1d = self._model.hidden.n_dim == 0

        return self

    def sample_and_weight(self, y, x):
        new_x = self._model.hidden.propagate(x)
        affine_transform = next(trans for trans in new_x.dist.transforms if isinstance(trans, AffineTransform))

        h_loc, h_scale = affine_transform.loc[:], affine_transform.scale[:]
        new_x.values = h_loc

        for _ in range(self._n_steps):
            h_loc.requires_grad_(True)

            y_dist = self._model.observable.build_density(new_x)

            logl = y_dist.log_prob(y) + new_x.dist.log_prob(h_loc)
            g = grad(logl, h_loc, grad_outputs=torch.ones_like(logl), create_graph=self._alpha is None)[-1]

            if self._alpha is None:
                step = -1 / grad(g, h_loc, grad_outputs=torch.ones_like(g))[-1]
                std = step.sqrt()
            else:
                std = h_scale.detach()
                step = self._alpha

            h_loc = h_loc.detach() + step * g.detach()
            new_x = new_x.copy(new_x.dist, h_loc)

        mean = h_loc.detach()

        if self._is1d:
            kernel = Normal(mean, std)
        else:
            kernel = Independent(Normal(mean, std), self._model.hidden.n_dim)

        new_x = new_x.copy(new_x.dist, kernel.sample())

        return new_x, self._weight_with_kernel(y, new_x, kernel)
