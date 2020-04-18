from .timeseries import StateSpaceModel, AffineProcess
import torch
from math import sqrt
from torch.distributions import Normal, MultivariateNormal, Independent
from .utils import construct_diag, TempOverride
from .module import Module, TensorContainer


def _propagate_sps(spx, spn, process, temp_params):
    """
    Propagate the Sigma points through the given process.
    :param spx: The state Sigma points
    :type spx: torch.Tensor
    :param spn: The noise Sigma points
    :type spn: torch.Tensor
    :param process: The process
    :type process: AffineProcess
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


# TODO: Rewrite this one to not save state
class UnscentedTransform(Module):
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
            x = torch.tensor(x)

        parts = x.shape if self._model.hidden_ndim < 1 else x.shape[:-self._model.hidden_ndim]

        self._mean = torch.zeros((*parts, self._ndim))
        self._cov = torch.zeros((*parts, self._ndim, self._ndim))
        self._sps = torch.zeros((*parts, 1 + 2 * self._ndim, self._ndim))

        # TODO: Perhaps move this to Timeseries?
        self._views = TensorContainer()
        shape = (parts[0], 1) if len(parts) > 0 else parts

        if len(parts) > 1:
            shape += (1,)

        for model in [self._model.hidden, self._model.observable]:
            params = tuple()
            for p in model.theta:
                if p.trainable:
                    view = p.view(*shape, *p.shape[1:])
                else:
                    view = p

                params += (view,)

            self._views.append(TensorContainer(*params))

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
        self._mean[..., self._sslc] = x if self._model.hidden_ndim > 0 else x.unsqueeze(-1)

        # ==== Set state covariance ===== #
        var = self._model.hidden.initial_dist.variance
        if self._model.hidden_ndim < 1:
            var.unsqueeze_(-1)

        self._cov[..., self._sslc, self._sslc] = construct_diag(var)

        # ==== Set noise covariance ===== #
        self._cov[..., self._hslc, self._hslc] = construct_diag(self._model.hidden.increment_dist.variance)
        self._cov[..., self._oslc, self._oslc] = construct_diag(self._model.observable.increment_dist.variance)

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

        spx = _propagate_sps(sps[..., self._sslc], sps[..., self._hslc], self._model.hidden, self._views[0])
        if only_x:
            return spx

        spy = _propagate_sps(spx, sps[..., self._oslc], self._model.observable, self._views[1])

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

        if self._model.hidden_ndim < 1:
            return Normal(self.xmean[..., 0], self.xcov[..., 0, 0].sqrt())

        return MultivariateNormal(self.xmean, scale_tril=torch.cholesky(self.xcov))

    @property
    def x_dist_indep(self):
        """
        Returns the current X-distribution but independent.
        :rtype: Normal|MultivariateNormal
        """

        if self._model.hidden_ndim < 1:
            return self.x_dist

        dist = Normal(self.xmean, self.xcov[..., self._diaginds, self._diaginds].sqrt())
        return Independent(dist, 1)

    @property
    def y_dist(self):
        """
        Returns the current Y-distribution.
        :rtype: Normal|MultivariateNormal
        """
        if self._model.obs_ndim < 1:
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

        temp = torch.matmul(ycov, gain.transpose(-1, -2))
        txcov = xcov - torch.matmul(gain, temp)

        return txmean, txcov, ymean, ycov