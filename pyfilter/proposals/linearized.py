from .base import Proposal
import torch
from torch.distributions import Normal, Independent
from ..timeseries import AffineProcess
from torch.autograd import grad


class Linearized(Proposal):
    def __init__(self, alpha=1e-3):
        """
        Implements a linearized proposal using Normal distributions. Do note that this proposal should be used for
        models that are log-concave in the observation density. Otherwise `Unscented` is more suitable.
        :param alpha: If `None`, uses second order information about the likelihood function, else takes step
        proportional to `alpha`.
        :type alpha: None|float
        """
        super().__init__()
        self._var_chooser = None
        self._alpha = alpha

    def set_model(self, model):
        if not (isinstance(model.observable, AffineProcess) and isinstance(model.hidden, AffineProcess)):
            raise ValueError(f'Both observable and hidden must be of type {AffineProcess.__class__.__name__}!')

        self._model = model
        self._var_chooser = (lambda u: u) if self._model.hidden_ndim < 2 else (lambda u: u._statevar)

        return self

    def construct(self, y, x):
        # ===== Mean of propagated dist ===== #
        h_loc, h_scale = self._model.hidden.mean_scale(x)
        h_loc.requires_grad_(True)

        # ===== Get gradients ===== #
        logl = self._model.observable.log_prob(y, h_loc) + self._model.hidden.log_prob(h_loc, x)

        var = self._var_chooser(h_loc)

        g = grad(logl, var, grad_outputs=torch.ones_like(logl), create_graph=self._alpha is None)[-1]

        # ===== Define mean and scale ===== #
        if self._alpha is None:
            var = -1 / grad(g, var, grad_outputs=torch.ones_like(g))[-1]
            std = var.sqrt()
            mean = h_loc.detach() + var * g.detach()

            x.detach_()
        else:
            std = h_scale.detach()
            mean = h_loc.detach() + self._alpha * g.detach()

        if self._model.hidden_ndim < 2:
            self._kernel = Normal(mean, std)
        else:
            self._kernel = Independent(Normal(mean, std), 1)

        return self