from ..timeseries.model import StateSpaceModel
import numpy as np
from .utils import outerm, expanddims, customcholesky, dot, mdot, outerv


def _propagate_sps(spx, spn, process):
    """
    Propagate the Sigma points through the given process.
    :param spx: The state Sigma points
    :param spn: The noise Sigma points
    :param process: The process
    :return:
    """
    mean = process.mean(spx)
    scale = process.scale(spx)

    if process.noise.ndim > 1:
        return mean + dot(scale, spn)

    return mean + scale * spn


def _helpweighter(a, b):
    """
    Performs a weighting along the second axis of `b` using `a` and sums.
    :param a: The weight array
    :param b: The array to weight.
    :return:
    """
    return np.einsum('i,ki...->k...', a, b)


def _covcalc(a, b, wc):
    """
    Calculates the covariance from a * b^t
    :param a:
    :param b:
    :return:
    """
    cov = np.einsum('ij...,kj...->jik...', a, b)

    return np.einsum('i,i...->...', wc, cov)


def _get_meancov(spxy, wm, wc):
    """
    Calculates the mean and covariance given sigma points for 2D processes.
    :param spxy: The state/observation sigma points
    :param wm: The W^m
    :param wc: The W^c
    :return:
    """

    x = _helpweighter(wm, spxy)
    centered = spxy - x[:, None, ...]

    return x, _covcalc(centered, centered, wc)


class UnscentedTransform(object):
    def __init__(self, model, a=1e-3, b=2, k=0):
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

        self._a = a
        self._b = b
        self._model = model
        self._ndim = 2 * model.hidden_ndim + model.obs_ndim
        self._lam = a ** 2 * (self._ndim + k) - self._ndim

    def _set_slices(self):
        """
        Sets the different slices for selecting states and noise.
        :return:
        """

        self._sslc = slice(self._model.hidden_ndim)
        self._hslc = slice(self._model.hidden_ndim, 2 * self._model.hidden_ndim)
        self._oslc = slice(2 * self._model.hidden_ndim, None)

        return self

    def _set_weights(self):
        """
        Generates the weights used for sigma point construction.
        :return:
        """

        self._wm = np.zeros(1 + 2 * self._ndim)
        self._wc = self._wm.copy()
        self._wm[0] = self._lam / (self._ndim + self._lam)
        self._wc[0] = self._wm[0] + (1 - self._a ** 2 + self._b)
        self._wm[1:] = self._wc[1:] = 1 / 2 / (self._ndim + self._lam)

        return self

    def _set_arrays(self, x):
        """
        Sets the mean and covariance arrays.
        :param x: The initial state.
        :return:
        """

        # ==== Define empty arrays ===== #

        parts = x.shape[1:] if self._model.hidden_ndim > 1 else x.shape

        self._mean = np.zeros((self._ndim, *parts))
        self._cov = np.zeros((self._ndim, self._ndim, *parts))
        self._sps = np.zeros((self._ndim, 1 + 2 * self._ndim, *parts))

        return self

    def initialize(self, x):
        """
        Initializes UnscentedTransform class.
        :param x: The initial values of the mean of the distribution.
        :type x: np.ndarray
        :return:
        """

        self._set_weights()._set_slices()._set_arrays(x)

        # ==== Set mean ===== #

        self._mean[self._sslc] = expanddims(self._model.hidden.i_mean(), x.ndim)

        # ==== Set state covariance ===== #
        scale = self._model.hidden.i_scale()
        if self._model.hidden_ndim > 1:
            self._cov[self._sslc, self._sslc] = expanddims(outerm(scale, scale), self._cov.ndim)
        else:
            self._cov[self._sslc, self._sslc] = scale ** 2

        # ==== Set noise covariance ===== #

        self._cov[self._hslc, self._hslc] = expanddims(self._model.hidden.noise.cov(), self._cov.ndim)
        self._cov[self._oslc, self._oslc] = expanddims(self._model.observable.noise.cov(), self._cov.ndim)

        return self

    def get_sps(self):
        """
        Constructs the Sigma points used for propagation.
        :return:
        """

        cholcov = np.sqrt(self._lam + self._ndim) * customcholesky(self._cov)

        self._sps[:, 0] = self._mean
        self._sps[:, 1:self._ndim+1] = self._mean[:, None] + cholcov
        self._sps[:, self._ndim+1:] = self._mean[:, None] - cholcov

        return self._sps

    def propagate_sps(self):
        """
        Propagate the Sigma points through the given process.
        :return:
        """

        sps = self.get_sps()

        spx = _propagate_sps(sps[self._sslc], sps[self._hslc], self._model.hidden)
        spy = _propagate_sps(spx, sps[self._oslc], self._model.observable)

        return spx, spy

    @property
    def xmean(self):
        """
        Returns the mean of the latest state.
        :return:
        """

        return self._mean[self._sslc]

    @property
    def xcov(self):
        """
        Returns the covariance of the latest state.
        :return:
        """

        return self._cov[self._sslc, self._sslc]

    def construct(self, y):
        """
        Constructs the mean and covariance given the current observation and previous state.
        :param y: The current observation
        :type y: np.ndarray
        :return:
        """

        # ==== Get mean and covariance ===== #

        txmean, txcov = self._get_m_and_p(y)

        # ==== Overwrite mean and covariance ==== #

        self._mean[self._sslc] = txmean
        self._cov[self._sslc, self._sslc] = txcov

        return txmean, txcov

    def _get_m_and_p(self, y):
        """
        Helper method for generating the mean and covariance.
        :param y: The latest observation
        :return:
        """

        # ==== Propagate Sigma points ==== #

        spx, spy = self.propagate_sps()

        # ==== Construct mean and covariance ==== #

        xmean, xcov = _get_meancov(spx, self._wm, self._wc)
        ymean, ycov = _get_meancov(spy, self._wm, self._wc)

        # ==== Calculate cross covariance ==== #

        xycov = _covcalc(spx - xmean[:, None], spy - ymean[:, None], self._wc)

        # ==== Calculate the gain ==== #

        gain = mdot(xycov, np.linalg.inv(ycov.T).T)

        # ===== Calculate true mean and covariance ==== #

        txmean = xmean + dot(gain, expanddims(y, ymean.ndim) - ymean)
        txcov = xcov - dot(gain, outerm(ycov, gain))

        return txmean, txcov

    def globalconstruct(self, y, x):
        """
        Constructs the mean and covariance given the current observation and previous state.
        :param y: The current observation
        :type y: np.ndarray
        :param x: The previous state
        :type x: np.ndarray
        :return:
        """

        # ==== Overwrite mean and covariance ==== #

        mean = expanddims(x.mean(axis=-1), x.ndim)
        centered = x - mean
        if self._model.hidden_ndim > 1:
            cov = expanddims(outerv(centered, centered).mean(axis=-1), x.ndim+1)
        else:
            cov = expanddims((centered ** 2).mean(axis=-1), x.ndim)

        self._mean[self._sslc] = mean
        self._cov[self._sslc, self._sslc] = cov

        # ==== Get mean and covariance ==== #

        txmean, txcov = self._get_m_and_p(y)

        return txmean, txcov