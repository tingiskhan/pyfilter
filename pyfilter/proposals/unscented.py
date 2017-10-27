from ..proposals import Linearized
import numpy as np
from ..utils.utils import dot, expanddims, mdot, choose
from ..distributions.continuous import MultivariateNormal, Normal


def _helpweighter(a, b):
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


class Unscented(Linearized):
    def __init__(self, model, *args, a=1, k=2, b=2, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._initialized = False

        self._mean = None
        self._cov = None
        self._totndim = 2 * self._model.hidden_ndim + self._model.obs_ndim

        # ==== Define helper variables ==== #
        self._a = a
        self._k = k
        self._b = b
        self._lam = self._a ** 2 * (self._totndim + self._k) - self._totndim

        # ==== Define UT weights ==== #
        self._wm = np.zeros(1 + 2 * self._totndim)
        self._wc = self._wm.copy()
        self._wm[0] = self._lam / (self._totndim + self._lam)
        self._wc[0] = self._wm[0] + (1 - self._a ** 2 + self._b)
        self._wm[1:] = self._wc[1:] = 1 / 2 / (self._totndim + self._lam)

        # ==== Helper slice variables ==== #
        self._stateslc = slice(self._model.hidden_ndim)
        self._hiddenslc = slice(self._model.hidden_ndim, 2*self._model.hidden_ndim)
        self._obsslc = slice(2*self._model.hidden_ndim, None)

    def _generate_sps(self):
        """
        Generates the sigma points using the current values of mean and covariance.
        :return:
        """

        cholled = np.linalg.cholesky((self._totndim + self._lam) * self._cov.T).T
        sps = np.zeros((self._totndim, 1 + 2 * self._totndim, *cholled.shape[2:]))

        sps[:, 0] = self._mean
        sps[:, 1:self._totndim+1] = self._mean[:, None, ...] + cholled
        sps[:, self._totndim+1:] = self._mean[:, None, ...] - cholled

        return sps

    def _initialize(self, x):
        """
        Initializes the proposal using first draw of the hidden state variables.
        :param x: The hidden state variable.
        :return:
        """

        parts = x.shape[1:] if self._model.hidden_ndim > 1 else x.shape

        self._mean = np.zeros((self._totndim, *parts))
        self._cov = np.zeros((self._totndim, self._totndim, *parts))

        self._mean[self._stateslc] = x

        if self._model.hidden_ndim < 2:
            self._cov[self._stateslc, self._stateslc] = self._model.hidden.i_scale() ** 2
        else:
            tscale = self._model.hidden.i_scale()
            cov = np.einsum('ij...,jk...->ik...', tscale, tscale)
            self._cov[self._stateslc, self._stateslc] = expanddims(cov, self._cov.ndim)

        self._cov[self._hiddenslc, self._hiddenslc] = expanddims(self._model.hidden.noise.cov(), self._cov.ndim)
        self._cov[self._obsslc, self._obsslc] = expanddims(self._model.observable.noise.cov(), self._cov.ndim)

        self._initialized = True

        return self

    def evalsp(self, spx, spn, process):
        """
        Calculates the sigma points for X.
        :param spx: The sigma points for x
        :param spn: THe sigma points for the noise
        :param process: The hidden/observable process
        :return:
        """

        mean = process.mean(spx)
        scale = process.scale(spx)

        if process.noise.ndim > 1:
            return mean + dot(scale, spn)

        return mean + scale * spn

    def draw(self, y, x, size=None, *args, **kwargs):
        x = self._meaner(x)
        if not self._initialized:
            self._initialize(x)

        sps = self._generate_sps()

        spx = self.evalsp(sps[self._stateslc], sps[self._hiddenslc], self._model.hidden)
        spy = self.evalsp(spx, sps[self._obsslc], self._model.observable)

        mx, px = _get_meancov(spx, self._wm, self._wc)
        my, py = _get_meancov(spy, self._wm, self._wc)

        pxy = _covcalc(spx - mx[:, None, ...], spy - my[:, None, ...], self._wc)

        gain = mdot(pxy, np.linalg.inv(py.T).T)

        xm = mx + dot(gain, expanddims(y, my.ndim) - my)
        temp = np.einsum('ij...,lj...->il...', py, gain)
        p = px - dot(gain, temp)

        if self._model.hidden_ndim > 1:
            self._kernel = MultivariateNormal(xm, np.linalg.cholesky(p.T).T)
        else:
            self._kernel = Normal(xm[0], np.sqrt(p[0, 0]))

        self._mean[self._stateslc] = xm
        self._cov[self._stateslc, self._stateslc] = p

        return self._kernel.rvs(size=size)