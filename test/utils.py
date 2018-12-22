import unittest
import pyfilter.utils.utils as helps
from scipy.stats import wishart
import numpy as np
from scipy.optimize import minimize
from time import time
from pyfilter.timeseries import BaseModel, Observable, StateSpaceModel
from torch.distributions import Normal, MultivariateNormal
from pyfilter.utils.unscentedtransform import UnscentedTransform
import torch


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return 0


def g0(alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x


def go(x, alpha, sigma):
    return sigma


def foo(x1, x2, alpha, sigma):
    return alpha * x1 + x2


def goo(x1, x2, alpha, sigma):
    return sigma


def fmvn(x, a, sigma):
    return x[0], x[1]


def f0mvn(a, sigma):
    return torch.zeros(2)


def fomvn(x, sigma):
    return x[0] + x[1] / 2


def gomvn(x, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_OuterProduct(self):
        a = np.random.normal(size=(3, 3))
        b = np.random.normal(size=(3, 3))

        true = a.dot(b.dot(a.T))
        est = helps.outer(a, b)

        assert np.allclose(true, est)

    def test_Dot(self):
        a = np.random.normal(size=(2, 2))
        b = np.random.normal(size=2)
        
        trueval = a.dot(b)

        est = helps.dot(a, b)

        assert np.allclose(trueval, est)

    def test_Outerv(self):
        a = np.random.normal(size=2)
        b = np.random.normal(size=2)

        trueval = a[:, None].dot(b[None, :])

        est = helps.outerv(a, b)

        assert np.allclose(est, trueval)

    def test_ExpandDims(self):
        a = np.random.normal(size=(3, 3))
        b = np.random.normal(size=(3, 3, 500, 500))

        newa = helps.expanddims(a, b.ndim)

        assert newa.shape == (3, 3, 1, 1)

    def test_mdot(self):
        a = np.random.normal(size=(3, 3))
        b = np.empty((*a.shape, 300, 300))
        b[:, :] = helps.expanddims(a, b.ndim)

        est = helps.mdot(a, b)

        assert np.allclose(est, helps.expanddims(a.dot(a), b.ndim))

    def test_CustomCholesky(self):
        cov = wishart(3, scale=np.eye(3)).rvs()

        extendedcov = np.empty((*cov.shape, 300, 300))
        extendedcov[:, :] = helps.expanddims(cov, extendedcov.ndim)

        choleskied = np.linalg.cholesky(cov)

        assert np.allclose(helps.expanddims(choleskied, extendedcov.ndim), helps.customcholesky(extendedcov))

    def test_Outerm(self):
        a = np.random.normal(size=(3, 3))
        b = np.random.normal(size=(3, 3))

        trueouter = a.dot(b.T)

        calcouter = helps.outerm(a, b)

        assert np.allclose(trueouter, calcouter)

    def test_BFGS(self):
        for i in range(500):
            x = np.random.normal()
            m = np.random.normal()

            func = lambda u: -np.exp(-(u - m) ** 2 / 2)

            trueanswer = minimize(func, x)
            approximate = helps.bfgs(func, x, tol=1e-8)

            assert (np.abs(m - approximate.x) < 1e-7)

    def test_BFGS_ParallellOptimization(self):
        x = np.random.normal(size=(1, 5000))
        m = np.random.normal()

        func = lambda u: -np.exp(-(u - m) ** 2 / 2)

        truestart = time()
        trueanswers = np.array([minimize(func, x[:, i]).x for i in range(x.shape[-1])])
        truetime = time() - truestart

        approxstart = time()
        approximate = helps.bfgs(func, x, tol=1e-7)
        approxtime = time() - approxstart

        print('naive: {:.3f}, parallel: {:.3f}, speedup: {:.2f}x'.format(truetime, approxtime, truetime / approxtime))

        assert (np.abs(approximate.x - m) < 1e-7).mean() > 0.95 and truetime / approxtime > 2

    def test_UnscentedTransform1D(self):
        # ===== 1D model ===== #
        norm = Normal(0., 1.)
        linear = BaseModel((f0, g0), (f, g), (1., 1.), (norm, norm))
        linearobs = Observable((fo, go), (1., 1.), norm)
        model = StateSpaceModel(linear, linearobs)

        # ===== Perform unscented transform ===== #
        x = model.hidden.i_sample(shape=3000)

        ut = UnscentedTransform(model).initialize(x).construct(0.)

        assert isinstance(ut.x_dist, Normal)

    def test_UnscentedTransform2D(self):
        # ===== 2D model ===== #
        mat = torch.eye(2)
        scale = torch.diag(mat)

        norm = Normal(0., 1.)
        mvn = MultivariateNormal(torch.zeros(2), torch.eye(2))
        mvnlinear = BaseModel((f0mvn, g0), (fmvn, g), (mat, scale), (mvn, mvn))
        mvnoblinear = Observable((fomvn, gomvn), (1.,), norm)

        mvnmodel = StateSpaceModel(mvnlinear, mvnoblinear)

        # ===== Perform unscented transform ===== #
        x = mvnmodel.hidden.i_sample()

        ut = UnscentedTransform(mvnmodel).initialize(x).construct(0.)

        assert isinstance(ut.x_dist, MultivariateNormal) and isinstance(ut.y_dist, Normal)