from .timeseries import StateSpaceModel, StochasticProcess
import torch
from math import sqrt
from torch.distributions import Normal, MultivariateNormal
from .utils import construct_diag, TempOverride
from .module import Module
from .timeseries.parameter import size_getter
from typing import Tuple, Union


def _propagate_sps(spx: torch.Tensor, spn: torch.Tensor, process: StochasticProcess,
                   temp_params: Tuple[torch.Tensor, ...]):
    """
    Propagate the Sigma points through the given process.
    :param spx: The state Sigma points
    :param spn: The noise Sigma points
    :param process: The process
    :return: Translated and scaled sigma points
    """

    is_md = process.ndim > 0

    if not is_md:
        spx = spx.squeeze(-1)
        spn = spn.squeeze(-1)

    with TempOverride(process, "_parameter_views", temp_params):
        out = process.propagate_u(spx, u=spn)
        return out if is_md else out.unsqueeze(-1)


def _covcalc(a: torch.Tensor, b: torch.Tensor, wc: torch.Tensor):
    """
    Calculates the covariance from a * b^t
    :param a: The `a` matrix
    :param b: The `b` matrix
    :return: The covariance
    """
    cov = a.unsqueeze(-1) * b.unsqueeze(-2)

    return (wc[:, None, None] * cov).sum(-3)


def _get_meancov(spxy: torch.Tensor, wm: torch.Tensor, wc: torch.Tensor):
    """
    Calculates the mean and covariance given sigma points for 2D processes.
    :param spxy: The state/observation sigma points
    :param wm: The W^m
    :param wc: The W^c
    :return: Mean and covariance
    """

    x = (wm.unsqueeze(-1) * spxy).sum(-2)
    centered = spxy - x.unsqueeze(-2)

    return x, _covcalc(centered, centered, wc)


class UFTCorrectionResult(object):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, state_slice: slice, ym: torch.Tensor, yc: torch.Tensor):
        self.ym = ym
        self.yc = yc

        self.mean = mean
        self.cov = cov
        self._sslc = state_slice

    @property
    def xm(self):
        return self.mean[..., self._sslc]

    @property
    def xc(self):
        return self.cov[..., self._sslc, self._sslc]

    @staticmethod
    def _helper(m, c):
        if m.shape[-1] > 1:
            return MultivariateNormal(m, c)

        return Normal(m[..., 0], c[..., 0, 0].sqrt())

    def x_dist(self):
        return self._helper(self.xm, self.xc)

    def y_dist(self):
        return self._helper(self.ym, self.yc)


class UFTPredictionResult(object):
    def __init__(self, spx: torch.Tensor, spy: torch.Tensor):
        self.spx = spx
        self.spy = spy


class AggregatedResult(object):
    def __init__(self, xm, xc, ym, yc):
        self.xm = xm
        self.xc = xc
        self.ym = ym
        self.yc = yc


class UnscentedFilterTransform(Module):
    def __init__(self, model: StateSpaceModel, a=1., b=2., k=0.):
        """
        Implements the Unscented Transform for a state space model.
        :param model: The model
        :param a: The alpha parameter. Defined on the interval [0, 1]
        :param b: The beta parameter. Optimal value for Gaussian models is 2
        :param k: The kappa parameter. To control the semi-definiteness
        """

        if len(model.hidden.increment_dist.event_shape) > 1:
            raise ValueError('Can at most handle vector valued processes!')

        if model.hidden._dist_builder or model.observable._dist_builder:
            raise ValueError('Cannot currently handle case when distribution is parameterized!')

        # ===== Model ===== #
        self._model = model
        self._trans_dim = 1 if len(model.hidden.increment_dist.event_shape) == 0 else \
            model.hidden.increment_dist.event_shape[0]

        self._ndim = model.hidden.num_vars + self._trans_dim + model.observable.num_vars

        # ===== Parameters =====#
        self._a = a
        self._b = b
        self._lam = a ** 2 * (self._ndim + k) - self._ndim

        # ===== Auxiliary variables ===== #
        self._views = None

        self._diaginds = range(model.hidden_ndim)

    def modules(self):
        return {}

    def _set_slices(self):
        """
        Sets the different slices for selecting states and noise.
        :return: Self
        """

        hidden_dim = self._model.hidden.num_vars

        self._sslc = slice(hidden_dim)
        self._hslc = slice(hidden_dim, hidden_dim + self._trans_dim)
        self._oslc = slice(hidden_dim + self._trans_dim, None)

        return self

    def _set_weights(self):
        """
        Generates the weights used for sigma point construction.
        :return: Self
        """

        self._wm = torch.zeros(1 + 2 * self._ndim)
        self._wc = self._wm.clone()
        self._wm[0] = self._lam / (self._ndim + self._lam)
        self._wc[0] = self._wm[0] + (1 - self._a ** 2 + self._b)
        self._wm[1:] = self._wc[1:] = 1 / 2 / (self._ndim + self._lam)

        return self

    def _set_arrays(self, shape: torch.Size):
        """
        Sets the mean and covariance arrays.
        :param shape: The shape
        :return: Self
        """

        view_shape = (shape[0], *(1 for _ in shape)) if len(shape) > 0 else shape
        self._views = self._model.viewify_params(view_shape, in_place=False)

        return self

    def initialize(self, shape: Union[int, Tuple[int, ...], torch.Size] = None):
        """
        Initializes UnscentedTransform class.
        :param shape: Shape of the state
        :type shape: int|tuple|torch.Size
        :return: Self
        """

        shape = size_getter(shape)
        self._set_weights()._set_slices()._set_arrays(shape)

        # ===== Define and covariance ===== #
        mean = torch.zeros((*shape, self._ndim))
        cov = torch.zeros((*shape, self._ndim, self._ndim))

        # ===== Set mean ===== #
        s_mean = self._model.hidden.i_sample((1000, *shape)).mean(0)
        if self._model.hidden_ndim < 1:
            s_mean.unsqueeze_(-1)

        mean[..., self._sslc] = s_mean

        # ==== Set state covariance ===== #
        var = s_cov = self._model.hidden.initial_dist.variance

        if self._model.hidden_ndim > 0:
            s_cov = construct_diag(var)

        cov[..., self._sslc, self._sslc] = s_cov

        # ==== Set noise covariance ===== #
        cov[..., self._hslc, self._hslc] = construct_diag(self._model.hidden.increment_dist.variance)
        cov[..., self._oslc, self._oslc] = construct_diag(self._model.observable.increment_dist.variance)

        return UFTCorrectionResult(mean, cov, self._sslc, None, None)

    def _get_sps(self, state: UFTCorrectionResult):
        """
        Constructs the Sigma points used for propagation.
        :return: Sigma points
        """

        cholcov = sqrt(self._lam + self._ndim) * torch.cholesky(state.cov)

        spx = state.mean.unsqueeze(-2)
        sph = state.mean[..., None, :] + cholcov
        spy = state.mean[..., None, :] - cholcov

        return torch.cat((spx, sph, spy), -2)

    def predict(self, utf_corr: UFTCorrectionResult):
        """
        Performs a prediction step using previous result from
        :param utf_corr: The correction result to use for predicting
        :return: Prediction
        """

        sps = self._get_sps(utf_corr)

        spx = _propagate_sps(sps[..., self._sslc], sps[..., self._hslc], self._model.hidden, self._views[0])
        spy = _propagate_sps(spx, sps[..., self._oslc], self._model.observable, self._views[1])

        return UFTPredictionResult(spx, spy)

    def calc_mean_cov(self, uft_pred: UFTPredictionResult):
        xmean, xcov = _get_meancov(uft_pred.spx, self._wm, self._wc)
        ymean, ycov = _get_meancov(uft_pred.spy, self._wm, self._wc)

        return AggregatedResult(xmean, xcov, ymean, ycov)

    def update_state(self, xm: torch.Tensor, xc: torch.Tensor, state: UFTCorrectionResult,
                     ym: torch.Tensor = None, yc: torch.Tensor = None):
        # ===== Overwrite ===== #
        mean = state.mean.clone()
        cov = state.cov.clone()

        mean[..., self._sslc] = xm
        cov[..., self._sslc, self._sslc] = xc

        return UFTCorrectionResult(mean, cov, self._sslc, ym, yc)

    def correct(self, y: torch.Tensor, uft_pred: UFTPredictionResult, prev_corr: UFTCorrectionResult):
        """
        Constructs the mean and covariance given the current observation and previous state.
        :param y: The current observation
        :param uft_pred: The prediction result to correct
        :param prev_corr: The previous correction result
        :return: Self
        """

        # ===== Calculate mean and covariance ====== #
        correction = self.calc_mean_cov(uft_pred)
        xmean, xcov, ymean, ycov = correction.xm, correction.xc, correction.ym, correction.yc

        # ==== Calculate cross covariance ==== #
        if xmean.dim() > 1:
            tx = uft_pred.spx - xmean.unsqueeze(-2)
        else:
            tx = uft_pred.spx - xmean

        if ymean.dim() > 1:
            ty = uft_pred.spy - ymean.unsqueeze(-2)
        else:
            ty = uft_pred.spy - ymean

        xycov = _covcalc(tx, ty, self._wc)

        # ==== Calculate the gain ==== #
        gain = torch.matmul(xycov, ycov.inverse())

        # ===== Calculate true mean and covariance ==== #
        txmean = xmean + torch.matmul(gain, (y - ymean).unsqueeze(-1))[..., 0]

        temp = torch.matmul(ycov, gain.transpose(-1, -2))
        txcov = xcov - torch.matmul(gain, temp)

        # ===== Overwrite ===== #
        return self.update_state(txmean, txcov, prev_corr, ymean, ycov)