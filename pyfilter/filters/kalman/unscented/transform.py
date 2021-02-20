import torch
from math import sqrt
from torch.nn import Module
from typing import Tuple
from ....utils import construct_diag_from_flat, size_getter
from ....typing import ShapeLike
from ....timeseries import StateSpaceModel, TimeseriesState
from ....parameter import ExtendedParameter
from .utils import propagate_sps, covariance
from .result import UFTCorrectionResult, UFTPredictionResult


class UnscentedFilterTransform(Module):
    MONTE_CARLO_ESTIMATES = 1000

    def __init__(self, model: StateSpaceModel, a=1.0, b=2.0, k=0.0):
        """
        Implements the Unscented Transform for a state space model.
        :param model: The model
        :param a: The alpha parameter. Defined on the interval [0, 1]
        :param b: The beta parameter. Optimal value for Gaussian models is 2
        :param k: The kappa parameter. To control the semi-definiteness
        """

        super().__init__()
        if len(model.hidden.increment_dist().event_shape) > 1:
            raise ValueError("Can at most handle vector valued processes!")

        if any(model.hidden.increment_dist.named_parameters()) or any(
            model.observable.increment_dist.named_parameters()
        ):
            raise ValueError("Cannot currently handle case when distribution is parameterized!")

        trans_dim = (
            1 if len(model.hidden.increment_dist().event_shape) == 0 else model.hidden.increment_dist().event_shape[0]
        )

        self._model = model
        self._n_dim = model.hidden.num_vars + trans_dim + model.observable.num_vars

        lam = a ** 2 * (self._n_dim + k) - self._n_dim
        self._cov_scale = sqrt(lam + self._n_dim)

        self._set_weights(a, b, lam)
        self._set_slices(trans_dim)

        self._view_shape = None
        self._diaginds = range(model.hidden_ndim)

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

    def initialize(self, shape: ShapeLike = None):
        shape = size_getter(shape)
        self._view_shape = (shape[0], *(1 for _ in shape)) if len(shape) > 0 else shape

        mean = torch.zeros((*shape, self._n_dim))
        cov = torch.zeros((*shape, self._n_dim, self._n_dim))

        initial_state = self._model.hidden.initial_sample((self.MONTE_CARLO_ESTIMATES, *shape))
        initial_state_mean = initial_state.state.mean(0)
        initial_state_var = initial_state.state.var(0)

        if self._model.hidden_ndim < 1:
            initial_state_mean.unsqueeze_(-1)

        mean[..., self._state_slc] = initial_state_mean

        initial_state.state = mean
        cov[..., self._state_slc, self._state_slc] = construct_diag_from_flat(
            initial_state_var, self._model.hidden_ndim
        )

        cov[..., self._hidden_slc, self._hidden_slc] = construct_diag_from_flat(
            self._model.hidden.increment_dist().variance, self._model.hidden_ndim
        )

        cov[..., self._obs_slc, self._obs_slc] = construct_diag_from_flat(
            self._model.observable.increment_dist().variance, self._model.obs_ndim
        )

        return UFTCorrectionResult(initial_state, cov, self._state_slc, None, None)

    @property
    def _hidden_views(self):
        return self._get_params_as_view(self._model.hidden)

    @property
    def _obs_views(self):
        return self._get_params_as_view(self._model.observable)

    def _get_params_as_view(self, module) -> Tuple[torch.Tensor, ...]:
        return tuple(
            p.view(self._view_shape) if isinstance(p, ExtendedParameter) else p for p in module.functional_parameters()
        )

    def update_state(
        self,
        xm: torch.Tensor,
        xc: torch.Tensor,
        state: TimeseriesState,
        prev_corr: UFTCorrectionResult,
        ym: torch.Tensor = None,
        yc: torch.Tensor = None,
    ):
        # ===== Overwrite ===== #
        mean = prev_corr.mean.state.clone()
        cov = prev_corr.cov.clone()

        mean[..., self._state_slc] = xm
        cov[..., self._state_slc, self._state_slc] = xc

        return UFTCorrectionResult(state.copy(mean), cov, self._state_slc, ym, yc)

    def predict(self, utf_corr: UFTCorrectionResult):
        sps = utf_corr.calculate_sigma_points(self._cov_scale)

        hidden_state = utf_corr.mean.copy(sps[..., self._state_slc])
        spx = propagate_sps(hidden_state, sps[..., self._hidden_slc], self._model.hidden, self._hidden_views)

        spy = propagate_sps(spx, sps[..., self._obs_slc], self._model.observable, self._obs_views)

        return UFTPredictionResult(spx, spy)

    def correct(self, y: torch.Tensor, uft_pred: UFTPredictionResult, prev_corr: UFTCorrectionResult):
        correction = uft_pred.get_mean_and_covariance(self._wm, self._wc)
        xmean, xcov, ymean, ycov = correction.xm, correction.xc, correction.ym, correction.yc

        if xmean.dim() > 1:
            tx = uft_pred.spx.state - xmean.unsqueeze(-2)
        else:
            tx = uft_pred.spx.state - xmean

        if ymean.dim() > 1:
            ty = uft_pred.spy.state - ymean.unsqueeze(-2)
        else:
            ty = uft_pred.spy.state - ymean

        xy_cov = covariance(tx, ty, self._wc)

        gain = torch.matmul(xy_cov, ycov.inverse())

        txmean = xmean + torch.matmul(gain, (y - ymean).unsqueeze(-1))[..., 0]

        temp = torch.matmul(ycov, gain.transpose(-1, -2))
        txcov = xcov - torch.matmul(gain, temp)

        return self.update_state(txmean, txcov, uft_pred.spx, prev_corr, ymean, ycov)
