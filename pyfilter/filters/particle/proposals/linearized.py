import torch
from torch.distributions import Normal, Independent, TransformedDistribution, AffineTransform
from torch.autograd import grad
from ....timeseries import AffineProcess
from .base import Proposal


class Linearized(Proposal):
    def __init__(self, alpha=0.25):
        """
        Implements a linearized proposal using Normal distributions. Do note that this proposal should be used for
        models that are log-concave in the observation density.
        :param alpha: If `None`, uses second order information about the likelihood function, else takes step
        proportional to `alpha`.
        :type alpha: None|float
        """
        super().__init__()
        self._alpha = alpha

    def set_model(self, model):
        if not (isinstance(model.observable, AffineProcess) and isinstance(model.hidden, AffineProcess)):
            raise ValueError(f"Both observable and hidden must be of type {AffineProcess.__class__.__name__}!")

        self._model = model

        return self

    def sample_and_weight(self, y, x):
        hidden_dist: TransformedDistribution = self._model.hidden.define_density(x)
        affine_transform = next(trans for trans in hidden_dist.transforms if isinstance(trans, AffineTransform))

        h_loc, h_scale = affine_transform.loc, affine_transform.scale
        h_loc.requires_grad_(True)

        state = self._model.hidden.propagate_state(h_loc, x)

        logl = self._model.observable.log_prob(y, state) + hidden_dist.log_prob(h_loc)
        g = grad(logl, h_loc, grad_outputs=torch.ones_like(logl), create_graph=self._alpha is None)[-1]

        if self._alpha is None:
            step = -1 / grad(g, h_loc, grad_outputs=torch.ones_like(g))[-1]
            std = step.sqrt()
        else:
            std = h_scale.detach()
            step = self._alpha

        h_loc.detach_()
        mean = h_loc.detach() + step * g.detach()

        if self._model.hidden_ndim == 0:
            kernel = Normal(mean, std)
        else:
            kernel = Independent(Normal(mean, std), self._model.hidden_ndim)

        new_x = self._model.hidden.propagate_state(kernel.sample(), x)

        return new_x, self._weight_with_kernel(y, new_x, hidden_dist, kernel)
