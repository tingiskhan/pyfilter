import torch
from .linear import LinearGaussianObservations
from ....timeseries import AffineProcess


class LocalLinearization(LinearGaussianObservations):
    """
    A proposal distribution useful for when the mean of the observable distribution is a non-linear function of the
    underlying state. The proposal linearizes the non-linear function around E[\mu(x_{t-1})] and then uses
    `LinearGaussianObservations` as a proposal.
    """

    def set_model(self, model):
        if model.obs_ndim > 0:
            raise Exception("This proposal distribution does not work for models having more observable dimension > 0!")

        if not (isinstance(model.observable, AffineProcess) and isinstance(model.hidden, AffineProcess)):
            raise ValueError(f"Both observable and hidden must be of type {AffineProcess.__class__.__name__}!")

        self._model = model

        return self

    def get_constant_and_offset(self, params, x):
        x.state.requires_grad_(True)

        loc, _ = self._model.observable.mean_scale(x)
        loc.backward(torch.ones_like(loc))
        grad_eval = x.state.grad

        x.state.detach_()
        loc = loc.detach()

        if self._model.hidden_ndim == 0:
            product = grad_eval * x.state
        else:
            product = (grad_eval * x.state).sum(-1)

        return grad_eval, loc - product
