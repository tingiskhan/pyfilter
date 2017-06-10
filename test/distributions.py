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
