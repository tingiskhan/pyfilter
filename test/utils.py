import unittest
import pyfilter.utils.utils as helps
from scipy.stats import wishart
import numpy as np


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