import unittest
import pyfilter.distributions.continuous as cont
import numpy as np
import scipy.stats as stats


class Tests(unittest.TestCase):
    def test_MVN(self):
        mean = np.zeros(3)
        cov = np.eye(3)

        mvn = cont.MultivariateNormal(mean, cov)

        true = stats.multivariate_normal.logpdf(mean, mean, cov)
        est = mvn.logpdf(mean)

        assert np.allclose(true, est)

    def test_MVNMultidimensional(self):
        mean = np.zeros(3)
        cov = np.eye(3)

        cov *= np.random.gamma(1, size=(3, 1))

        mvn = cont.MultivariateNormal(mean, cov)

        choleskied = np.linalg.cholesky(cov)

        true = stats.multivariate_normal.logpdf(mean, mean, choleskied ** 2)

        expanded_mean = np.zeros((3, 50, 50))
        expanded_mean[:, :, :] = mean[:, None, None]
        expanded_cov = np.zeros((3, 3, 50, 50))
        expanded_cov[:, :, :, :] = choleskied[:, :, None, None]

        est = mvn.logpdf(mean, expanded_mean, expanded_cov)

        assert np.allclose(true, est)
