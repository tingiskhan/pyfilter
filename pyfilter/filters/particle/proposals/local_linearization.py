import torch
from .linear import LinearGaussianObservations
from ....timeseries import AffineProcess


class LocalLinearization(LinearGaussianObservations):
    """
    A proposal distribution useful for when the mean of the observable distribution is a non-linear function of the
    underlying state. The proposal linearizes the non-linear function around E[\mu(x_{t-1})] and then uses
    `LinearGaussianObservations` as a proposal.
    """

    def __init__(self):
        super().__init__()

        self._hidden_is1d = None
        self._observable_is1d = None

    def set_model(self, model):
        if model.observable.n_dim > 0:
            raise Exception("This proposal distribution does not work for models having more observable dimension > 0!")

        if not (isinstance(model.observable, AffineProcess) and isinstance(model.hidden, AffineProcess)):
            raise ValueError(f"Both observable and hidden must be of type {AffineProcess.__class__.__name__}!")

        self._model = model

        self._hidden_is1d = self._model.hidden.n_dim == 0
        self._observable_is1d = self._model.observable.n_dim == 0

        return self

    # TODO: Fix
    def get_constant_and_offset(self, params, x):
        x.values.requires_grad_(True)

        loc, _ = self._model.observable.mean_scale(x)
        loc.backward(torch.ones_like(loc))
        grad_eval = x.values.grad

        x.values.detach_()
        loc = loc.detach()

        product = grad_eval * x.values
        if not self._hidden_is1d:
            product = product.sum(-1)

        return grad_eval, loc - product
