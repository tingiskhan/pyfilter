import torch
from math import sqrt
from torch.nn import Module
from typing import Tuple
from torch.distributions import MultivariateNormal, Normal
from .utils import propagate_sps, covariance, get_bad_inds, _EPS
from .result import UFTCorrectionResult, UFTPredictionResult
from ....utils import construct_diag_from_flat, size_getter
from ....typing import ShapeLike
from ....timeseries import StateSpaceModel, NewState
from ....parameter import ExtendedParameter


class UnscentedFilterTransform(Module):
    """
    Implements the Unscented Transform for a state space model.
    """

    MONTE_CARLO_ESTIMATES = 10_000

    def __init__(self, model: StateSpaceModel, alpha=1.0, beta=2.0, kappa=0.0):
        """
        Initializes the ``UnscentedFilterTransform``.

        Args:
            model: The state space model for which to filter.
            alpha: Defined on the interval [0, 1].
            beta: Optimal value for Gaussian models is 2.
            kappa: Controls the semi-definiteness of the covariance of the sigma points.
        """

        super().__init__()
        if model.hidden.n_dim > 1:
            raise ValueError("Can at most handle vector valued processes!")

        if any(model.hidden.increment_dist.named_parameters()) or any(
            model.observable.increment_dist.named_parameters()
        ):
            raise ValueError("Cannot currently handle case when distribution is parameterized!")

        trans_dim = 1 if model.hidden.n_dim == 0 else model.hidden.increment_dist().event_shape[0]

        self._model = model
        self._n_dim = model.hidden.num_vars + trans_dim + model.observable.num_vars

        lam = alpha ** 2 * (self._n_dim + kappa) - self._n_dim
        self._cov_scale = sqrt(lam + self._n_dim)

        self._set_weights(alpha, beta, lam)
        self._set_slices(trans_dim)

        self._view_shape = None

    def _set_slices(self, trans_dim):
        hidden_dim = self._model.hidden.num_vars

        self._state_slc = slice(hidden_dim)
        self._hidden_slc = slice(hidden_dim, hidden_dim + trans_dim)
        self._obs_slc = slice(hidden_dim + trans_dim, None)

    def _set_weights(self, a, b, lamda):
        temp = torch.zeros(1 + 2 * self._n_dim)

        self.register_buffer("_wm", temp)
        self.register_buffer("_wc", temp.clone())

        self._wm[0] = lamda / (self._n_dim + lamda)
        self._wc[0] = self._wm[0] + (1 - a ** 2 + b)
        self._wm[1:] = self._wc[1:] = 1 / 2 / (self._n_dim + lamda)

    def initialize(self, shape: ShapeLike = None) -> UFTCorrectionResult:
        shape = size_getter(shape)
        self._view_shape = (shape[0], *(1 for _ in shape)) if len(shape) > 0 else shape

        mean = torch.zeros((*shape, self._n_dim), device=self._wm.device)
        cov = torch.zeros((*shape, self._n_dim, self._n_dim), device=self._wm.device)

        initial_state = self._model.hidden.initial_sample((self.MONTE_CARLO_ESTIMATES, *shape))
        initial_state_mean = initial_state.values.mean(0)
        initial_state_var = initial_state.values.var(0)

        if self._model.hidden.n_dim < 1:
            initial_state_mean.unsqueeze_(-1)

        mean[..., self._state_slc] = initial_state_mean
        cov[..., self._state_slc, self._state_slc] = construct_diag_from_flat(
            initial_state_var, self._model.hidden.n_dim
        )

        cov[..., self._hidden_slc, self._hidden_slc] = construct_diag_from_flat(
            self._model.hidden.increment_dist().sample((self.MONTE_CARLO_ESTIMATES,)).var(0), self._model.hidden.n_dim
        )

        cov[..., self._obs_slc, self._obs_slc] = construct_diag_from_flat(
            self._model.observable.increment_dist().sample((self.MONTE_CARLO_ESTIMATES,)).var(0),
            self._model.observable.n_dim,
        )

        dist = MultivariateNormal(loc=mean, covariance_matrix=cov, validate_args=False)

        return UFTCorrectionResult(initial_state.copy(dist=dist), self._state_slc, None)

    @property
    def _hidden_views(self):
        return self._get_params_as_view(self._model.hidden)

    @property
    def _obs_views(self):
        return self._get_params_as_view(self._model.observable)

    def _get_params_as_view(self, module) -> Tuple[torch.Tensor, ...]:
        return module.functional_parameters(
            f=lambda p: p.view(*self._view_shape, *p.shape[1:]) if isinstance(p, ExtendedParameter) else p
        )

    def update_state(
        self,
        xm: torch.Tensor,
        xc: torch.Tensor,
        x_state: NewState,
        prev_corr: UFTCorrectionResult,
        ym: torch.Tensor = None,
        yc: torch.Tensor = None,
        y_state: NewState = None,
    ) -> UFTCorrectionResult:
        mean = prev_corr.x.dist.mean.clone()
        cov = prev_corr.x.dist.covariance_matrix.clone()

        # We only allow fixing bad matrices when we're running batched mode
        if cov.dim() > 2:
            bad_matrix = get_bad_inds(xc, yc)

            if bad_matrix.all():
                raise Exception("All batches were singular or nan!")

            xc[bad_matrix, self._state_slc, self._state_slc] = _EPS * torch.eye(xc.shape[-1], device=xc.device)
            xm[bad_matrix, self._state_slc] = 0.0

            yc[bad_matrix] = _EPS * torch.eye(yc.shape[-1], device=yc.device)
            ym[bad_matrix] = 0.0

        mean[..., self._state_slc] = xm
        cov[..., self._state_slc, self._state_slc] = xc

        x_state = x_state.copy(dist=MultivariateNormal(loc=mean, covariance_matrix=cov, validate_args=False))

        if self._model.observable.n_dim > 0:
            y_state = y_state.copy(dist=MultivariateNormal(loc=ym, covariance_matrix=yc, validate_args=False))
        else:
            y_state = y_state.copy(dist=Normal(loc=ym[..., 0], scale=yc[..., 0, 0].sqrt(), validate_args=False))

        return UFTCorrectionResult(x_state, self._state_slc, y_state)

    def predict(self, utf_corr: UFTCorrectionResult) -> UFTPredictionResult:
        sps = utf_corr.calculate_sigma_points(self._cov_scale)

        hidden_state = utf_corr.x.copy(values=sps[..., self._state_slc])
        spx = propagate_sps(hidden_state, sps[..., self._hidden_slc], self._model.hidden, self._hidden_views)

        spy = propagate_sps(spx, sps[..., self._obs_slc], self._model.observable, self._obs_views)

        return UFTPredictionResult(spx, spy)

    def correct(self, y: torch.Tensor, uft_pred: UFTPredictionResult, prev_corr: UFTCorrectionResult) -> UFTCorrectionResult:
        (x_m, x_c), (y_m, y_c) = uft_pred.get_mean_and_covariance(self._wm, self._wc)

        if x_m.dim() > 1:
            tx = uft_pred.spx.values - x_m.unsqueeze(-2)
        else:
            tx = uft_pred.spx.values - x_m

        if y_m.dim() > 1:
            ty = uft_pred.spy.values - y_m.unsqueeze(-2)
        else:
            ty = uft_pred.spy.values - y_m

        xy_cov = covariance(tx, ty, self._wc)

        gain = torch.matmul(xy_cov, y_c.inverse())

        txmean = x_m + torch.matmul(gain, (y - y_m).unsqueeze(-1))[..., 0]

        temp = torch.matmul(y_c, gain.transpose(-1, -2))
        txcov = x_c - torch.matmul(gain, temp)

        return self.update_state(txmean, txcov, uft_pred.spx, prev_corr, y_m, y_c, uft_pred.spy)
