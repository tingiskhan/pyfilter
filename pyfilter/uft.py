from .timeseries import StateSpaceModel
from .timeseries.base import StochasticProcess
import torch
from math import sqrt
from torch.distributions import Normal, MultivariateNormal
from .utils import construct_diag, TempOverride
from .module import Module, TensorContainer
from .timeseries.parameter import size_getter


def _propagate_sps(spx, spn, process, temp_params):
    """
    Propagate the Sigma points through the given process.
    :param spx: The state Sigma points
    :type spx: torch.Tensor
    :param spn: The noise Sigma points
    :type spn: torch.Tensor
    :param process: The process
    :type process: StochasticProcess
    :return: Translated and scaled sigma points
    :rtype: torch.Tensor
    """

    is_md = process.ndim > 0

    if not is_md:
        spx = spx.squeeze(-1)
        spn = spn.squeeze(-1)

    with TempOverride(process, '_theta_vals', temp_params):
        out = process.propagate_u(spx, u=spn)
        return out if is_md else out.unsqueeze(-1)


def _covcalc(a, b, wc):
    """
    Calculates the covariance from a * b^t
    :param a: The `a` matrix
    :type a: torch.Tensor
    :param b: The `b` matrix
    :type b: torch.Tensor
    :return: The covariance
    :rtype: torch.Tensor
    """
    cov = a.unsqueeze(-1) * b.unsqueeze(-2)

    return (wc[:, None, None] * cov).sum(-3)


def _get_meancov(spxy, wm, wc):
    """
    Calculates the mean and covariance given sigma points for 2D processes.
    :param spxy: The state/observation sigma points
    :type spxy: torch.Tensor
    :param wm: The W^m
    :type wm: torch.Tensor
    :param wc: The W^c
    :type wc: torch.Tensor
    :return: Mean and covariance
    :rtype: tuple of torch.Tensor
    """

    x = (wm.unsqueeze(-1) * spxy).sum(-2)
    centered = spxy - x.unsqueeze(-2)

    return x, _covcalc(centered, centered, wc)


class UFTCorrectionResult(object):
    def __init__(self, xm, xc, ym, yc):
        self.xm = xm
        self.xc = xc

        self.ym = ym
        self.yc = yc

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
    def __init__(self, spx, spy):
        self.spx = spx
        self.spy = spy


# TODO: Rewrite this one to not save state
class UnscentedFilterTransform(Module):
    def __init__(self, model, a=1, b=2, k=0):
        """
        Implements the Unscented Transform for a state space model.
        :param model: The model
        :type model: StateSpaceModel
        :param a: The alpha parameter. Defined on the interval [0, 1]
        :type a: float
        :param b: The beta parameter. Optimal value for Gaussian models is 2
        :type b: float
        :param k: The kappa parameter. To control the semi-definiteness
        :type k: float
        """

        if len(model.hidden.increment_dist.event_shape) > 1:
            raise ValueError('Can at most handle vector valued processes!')

        if model.hidden.distributional_theta or model.observable.distributional_theta:
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
        self._ymean = None
        self._ycov = None
        self._views = None

        self._diaginds = range(model.hidden_ndim)

    def modules(self):
        return {}

    def _set_slices(self):
        """
        Sets the different slices for selecting states and noise.
        :return: Instance of self
        :rtype: UnscentedTransform
        """

        hidden_dim = self._model.hidden.num_vars

        self._sslc = slice(hidden_dim)
        self._hslc = slice(hidden_dim, hidden_dim + self._trans_dim)
        self._oslc = slice(hidden_dim + self._trans_dim, None)

        return self

    def _set_weights(self):
        """
        Generates the weights used for sigma point construction.
        :return: Instance of self
        :rtype: UnscentedTransform
        """

        self._wm = torch.zeros(1 + 2 * self._ndim)
        self._wc = self._wm.clone()
        self._wm[0] = self._lam / (self._ndim + self._lam)
        self._wc[0] = self._wm[0] + (1 - self._a ** 2 + self._b)
        self._wm[1:] = self._wc[1:] = 1 / 2 / (self._ndim + self._lam)

        return self

    def _set_arrays(self, shape):
        """
        Sets the mean and covariance arrays.
        :param shape: The shape
        :type shape: torch.Shape
        :return: Instance of self
        :rtype: UnscentedTransform
        """

        # ==== Define empty arrays ===== #
        self._mean = torch.zeros((*shape, self._ndim))
        self._cov = torch.zeros((*shape, self._ndim, self._ndim))

        # TODO: Perhaps move this to Timeseries?
        self._views = TensorContainer()
        view_shape = (shape[0], 1) if len(shape) > 0 else shape

        for model in [self._model.hidden, self._model.observable]:
            params = tuple()
            for p in model.theta:
                if p.trainable:
                    view = p.view(*view_shape, *p.shape[1:])
                else:
                    view = p

                params += (view,)

            self._views.append(TensorContainer(*params))

        return self

    def initialize(self, shape=None):
        """
        Initializes UnscentedTransform class.
        :param shape: Shape of the state
        :type shape: int|tuple|torch.Size
        :return: Instance of self
        :rtype: UnscentedTransform
        """

        self._set_weights()._set_slices()._set_arrays(size_getter(shape))

        # ==== Set mean ===== #
        self._mean[..., self._sslc] = self._model.hidden.i_sample(1000).mean(0)

        # ==== Set state covariance ===== #
        var = cov = self._model.hidden.initial_dist.variance

        if self._model.hidden_ndim > 0:
            cov = construct_diag(var)

        self._cov[..., self._sslc, self._sslc] = cov

        # ==== Set noise covariance ===== #
        self._cov[..., self._hslc, self._hslc] = construct_diag(self._model.hidden.increment_dist.variance)
        self._cov[..., self._oslc, self._oslc] = construct_diag(self._model.observable.increment_dist.variance)

        return UFTCorrectionResult(self._mean[..., self._sslc], self._cov[..., self._sslc, self._sslc], None, None)

    def _get_sps(self, mean, cov):
        """
        Constructs the Sigma points used for propagation.
        :return: Sigma points
        :rtype: torch.Tensor
        """

        self._mean[..., self._sslc] = mean

        # if self._model.hidden_ndim > 0:
        #     cov = construct_diag(cov)

        self._cov[..., self._sslc, self._sslc] = cov

        cholcov = sqrt(self._lam + self._ndim) * torch.cholesky(self._cov)

        spx = self._mean.unsqueeze(-2)
        sph = self._mean[..., None, :] + cholcov
        spy = self._mean[..., None, :] - cholcov

        return torch.cat((spx, sph, spy), -2)

    def predict(self, utf):
        """
        Performs a prediction step using previous result from
        :param utf:
        :type utf: UFTCorrectionResult
        :return:
        """

        # ===== Propagate sigma points ===== #
        sps = self._get_sps(utf.xm, utf.xc)

        spx = _propagate_sps(sps[..., self._sslc], sps[..., self._hslc], self._model.hidden, self._views[0])
        spy = _propagate_sps(spx, sps[..., self._oslc], self._model.observable, self._views[1])

        return UFTPredictionResult(spx, spy)

    def calc_mean_cov(self, uft_pred):
        xmean, xcov = _get_meancov(uft_pred.spx, self._wm, self._wc)
        ymean, ycov = _get_meancov(uft_pred.spy, self._wm, self._wc)

        return (xmean, xcov), (ymean, ycov)

    def correct(self, y, uft_pred):
        """
        Constructs the mean and covariance given the current observation and previous state.
        :param y: The current observation
        :type y: torch.Tensor
        :param uft_pred:
        :type uft_pred: UFTPredictionResult
        :return: Self
        :rtype: UnscentedTransform
        """

        # ===== Calculate mean and covariance ====== #
        (xmean, xcov), (ymean, ycov) = self.calc_mean_cov(uft_pred)

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
        txcov = ycov - torch.matmul(gain, temp)

        return UFTCorrectionResult(txmean, txcov, ymean, ycov)