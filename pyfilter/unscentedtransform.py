from .timeseries import StateSpaceModel, BaseModel
import numpy as np
import torch
from math import sqrt
from torch.distributions import Normal, MultivariateNormal, Independent
from .utils import construct_diag


def _propagate_sps(spx, spn, process):
    """
    Propagate the Sigma points through the given process.
    :param spx: The state Sigma points
    :type spx: torch.Tensor
    :param spn: The noise Sigma points
    :type spn: torch.Tensor
    :param process: The process
    :type process: BaseModel
    :return: Translated and scaled sigma points
    :rtype: torch.Tensor
    """
    mean = process.mean(spx)
    scale = process.scale(spx)

    if process.ndim < 2:
        if mean.dim() < spn.dim():
            mean = mean.unsqueeze(-1)

        return mean + scale * spn

    return mean + scale.unsqueeze(-2) * spn


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
    cov = torch.einsum('...ji,...jk->...jik', a, b)

    return torch.einsum('i,...ijk->...jk', wc, cov)


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


class UnscentedTransform(object):
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

        self._a = a
        self._b = b
        self._model = model
        self._ndim = 2 * model.hidden_ndim + model.obs_ndim
        self._lam = a ** 2 * (self._ndim + k) - self._ndim
        self._ymean = None
        self._ycov = None

        self._initialized = False
        self._diaginds = range(model.hidden_ndim)

    @property
    def initialized(self):
        """
        Returns boolean indicating whether it is initialized or not.
        :rtype: bool
        """

        return self._initialized

    def _set_slices(self):
        """
        Sets the different slices for selecting states and noise.
        :return: Instance of self
        :rtype: UnscentedTransform
        """

        self._sslc = slice(self._model.hidden_ndim)
        self._hslc = slice(self._model.hidden_ndim, 2 * self._model.hidden_ndim)
        self._oslc = slice(2 * self._model.hidden_ndim, None)

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

    def _set_arrays(self, x):
        """
        Sets the mean and covariance arrays.
        :param x: The initial state.
        :type x: torch.Tensor
        :return: Instance of self
        :rtype: UnscentedTransform
        """

        # ==== Define empty arrays ===== #
        if not isinstance(x, torch.Tensor):
            x = np.array(x)

        parts = x.shape[:-1] if self._model.hidden_ndim > 1 else x.shape

        self._mean = torch.zeros((*parts, self._ndim))
        self._cov = torch.zeros((*parts, self._ndim, self._ndim))
        self._sps = torch.zeros((*parts, 1 + 2 * self._ndim, self._ndim))

        return self

    def initialize(self, x):
        """
        Initializes UnscentedTransform class.
        :param x: The initial values of the mean of the distribution.
        :type x: torch.Tensor
        :return: Instance of self
        :rtype: UnscentedTransform
        """

        self._set_weights()._set_slices()._set_arrays(x)

        # ==== Set mean ===== #
        self._mean[..., self._sslc] = x if self._model.hidden_ndim > 1 else x.unsqueeze(-1)

        # ==== Set state covariance ===== #
        self._cov[..., self._sslc, self._sslc] = construct_diag(self._model.hidden.i_scale() ** 2)

        # ==== Set noise covariance ===== #
        self._cov[..., self._hslc, self._hslc] = construct_diag(self._model.hidden.noise.variance)
        self._cov[..., self._oslc, self._oslc] = construct_diag(self._model.observable.noise.variance)

        self._initialized = True

        return self

    def get_sps(self):
        """
        Constructs the Sigma points used for propagation.
        :return: Sigma points
        :rtype: torch.Tensor
        """
        cholcov = sqrt(self._lam + self._ndim) * torch.cholesky(self._cov)

        self._sps[..., 0, :] = self._mean
        self._sps[..., 1:self._ndim+1, :] = self._mean[..., None, :] + cholcov
        self._sps[..., self._ndim+1:, :] = self._mean[..., None, :] - cholcov

        return self._sps

    def propagate_sps(self, only_x=False):
        """
        Propagate the Sigma points through the given process.
        :return: Sigma points of x and y
        :rtype: tuple of torch.Tensor
        """

        sps = self.get_sps()

        spx = _propagate_sps(sps[..., self._sslc], sps[..., self._hslc], self._model.hidden)
        if only_x:
            return spx

        spy = _propagate_sps(spx, sps[..., self._oslc], self._model.observable)

        return spx, spy

    @property
    def xmean(self):
        """
        Returns the mean of the latest state.
        :return: The mean of state
        :rtype: torch.Tensor
        """

        return self._mean[..., self._sslc].clone()

    @xmean.setter
    def xmean(self, x):
        """
        Sets the mean of the latest state.
        :param x: The mean state to use for overriding
        :type x: torch.Tensor
        """

        self._mean[..., self._sslc] = x

    @property
    def xcov(self):
        """
        Returns the covariance of the latest state.
        :return: The state covariance
        :rtype: torch.Tensor
        """

        return self._cov[..., self._sslc, self._sslc]

    @xcov.setter
    def xcov(self, x):
        """
        Sets the covariance of the latest state
        :param x: The state covariance to use for overriding
        :type x: torch.Tensor
        """

        self._cov[..., self._sslc, self._sslc] = x

    @property
    def ymean(self):
        """
        Returns the mean of the observation.
        :return: The mean of the observational process
        :rtype: torch.Tensor
        """

        return self._ymean

    @property
    def ycov(self):
        """
        Returns the covariance of the observation.
        :return: The covariance of the observational process
        :rtype: torch.Tensor
        """

        return self._ycov

    @property
    def x_dist(self):
        """
        Returns the current X-distribution.
        :rtype: Normal|MultivariateNormal
        """

        if self._model.hidden_ndim < 2:
            return Normal(self.xmean[..., 0], self.xcov[..., 0, 0].sqrt())

        return MultivariateNormal(self.xmean, scale_tril=torch.cholesky(self.xcov))

    @property
    def x_dist_indep(self):
        """
        Returns the current X-distribution but independent.
        :rtype: Normal|MultivariateNormal
        """

        if self._model.hidden_ndim < 2:
            return self.x_dist

        return Independent(Normal(self.xmean, self.xcov[..., self._diaginds, self._diaginds].sqrt()), -1)

    @property
    def y_dist(self):
        """
        Returns the current Y-distribution.
        :rtype: Normal|MultivariateNormal
        """
        if self._model.obs_ndim < 2:
            return Normal(self.ymean[..., 0], self.ycov[..., 0, 0].sqrt())

        return MultivariateNormal(self.ymean, scale_tril=torch.cholesky(self.ycov))

    def construct(self, y):
        """
        Constructs the mean and covariance given the current observation and previous state.
        :param y: The current observation
        :type y: torch.Tensor|float
        :return: Self
        :rtype: UnscentedTransform
        """

        # ==== Get mean and covariance ===== #

        txmean, txcov, ymean, ycov = self._get_m_and_p(y)

        # ==== Overwrite mean and covariance ==== #

        self._ymean = ymean
        self._ycov = ycov
        self._mean[..., self._sslc] = txmean
        self._cov[..., self._sslc, self._sslc] = txcov

        return self

    def get_meancov(self):
        """
        Constructs the mean and covariance for the hidden and observable process respectively.
        :return: The mean and covariance
        :rtype: tuple
        """

        # ==== Propagate Sigma points ==== #

        spx, spy = self.propagate_sps()

        # ==== Construct mean and covariance ==== #

        xmean, xcov = _get_meancov(spx, self._wm, self._wc)
        ymean, ycov = _get_meancov(spy, self._wm, self._wc)

        return (xmean, xcov, spx), (ymean, ycov, spy)

    def _get_m_and_p(self, y):
        """
        Helper method for generating the mean and covariance.
        :param y: The latest observation
        :type y: float|torch.Tensor
        :return: The estimated mean and covariances of state and observation
        :rtype: tuple of torch.Tensor
        """

        (xmean, xcov, spx), (ymean, ycov, spy) = self.get_meancov()

        # ==== Calculate cross covariance ==== #
        if xmean.dim() > 1:
            tx = spx - xmean.unsqueeze(-2)
        else:
            tx = spx - xmean

        if ymean.dim() > 1:
            ty = spy - ymean.unsqueeze(-2)
        else:
            ty = spy - ymean

        xycov = _covcalc(tx, ty, self._wc)

        # ==== Calculate the gain ==== #
        gain = torch.matmul(xycov, ycov.inverse())

        # ===== Calculate true mean and covariance ==== #
        txmean = xmean + torch.matmul(gain, (y - ymean).unsqueeze(-1))[..., 0]

        temp = torch.einsum('...ij,...kj->...ik', (ycov, gain))
        txcov = xcov - torch.matmul(gain, temp)

        return txmean, txcov, ymean, ycov