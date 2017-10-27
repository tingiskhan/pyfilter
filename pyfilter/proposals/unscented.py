from ..proposals import Linearized
import numpy as np
from ..utils.utils import dot, expanddims, mdot
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
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._initialized = False

        self._mean = None
        self._cov = None
        self._totndim = 2 * self._model.hidden_ndim + self._model.obs_ndim

        self._a = 1
        self._k = 2
        self._b = 0
        self._lam = self._a ** 2 * (self._totndim + self._k) - self._totndim

        self._wm = np.zeros(1 + 2 * self._totndim)
        self._wc = self._wm.copy()
        self._wm[0] = self._lam / (self._totndim + self._lam)
        self._wm[1:] = self._wc[1:] = self._wm[0] / 2

        self._wc[0] = self._wm[0] + (1 - self._a ** 2 + self._b)

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
            cov = mdot(tscale.transpose((1, 0)), tscale)
            self._cov[self._stateslc, self._stateslc] = expanddims(cov, self._cov.ndim)

        self._cov[self._hiddenslc, self._hiddenslc] = expanddims(self._model.hidden.noise.cov(), self._cov.ndim)
        self._cov[self._obsslc, self._obsslc] = expanddims(self._model.observable.noise.cov(), self._cov.ndim)

        return self

    def get_spx(self, sp, process, slc):
        """
        Calculates the sigma points for X.
        :param sp: The sigma points
        :param process: The hidden/observable process
        :param slc: Which slice to use
        :return:
        """

        mean = process.mean(sp[self._stateslc])
        scale = process.scale(sp[self._stateslc])

        if process.noise.ndim > 1:
            return mean + dot(scale, sp[slc])

        return mean + scale * sp[slc]

    def draw(self, y, x, size=None, *args, **kwargs):
        x = self._meaner(x)
        if not self._initialized:
            self._initialize(x)

        sps = self._generate_sps()

        spx = self.get_spx(sps, self._model.hidden, self._hiddenslc)
        spy = self.get_spx(sps, self._model.observable, self._obsslc)

        mx, px = _get_meancov(spx, self._wm, self._wc)
        my, py = _get_meancov(spy, self._wm, self._wc)

        pxy = _covcalc(spx - mx[:, None, ...], spy - my[:, None, ...], self._wc)

        gain = np.einsum('ij...,jl...->il...', pxy, np.linalg.inv(py.T).T)

        xm = mx + dot(gain, y - my)
        temp = np.einsum('ij...,lj...->il...', py, gain)
        p = px - dot(gain, temp)

        if self._model.hidden_ndim > 1:
            self._kernel = MultivariateNormal(xm, np.linalg.cholesky(p.T).T)
        else:
            self._kernel = Normal(xm[0], np.sqrt(p[0, 0]))

        self._mean[self._stateslc] = xm
        self._cov[self._stateslc, self._stateslc] = p

        return self._kernel.rvs(size=size)

