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
        mean = np.random.normal(size=3)
        cov = stats.wishart(3, scale=np.eye(3)).rvs()

        choleskied = np.linalg.cholesky(cov)

        mvn = cont.MultivariateNormal(mean, cov)

        eps = np.random.uniform(high=0.25)
        true = stats.multivariate_normal.logpdf(mean, mean + eps, cov)

        expanded_mean = np.zeros((3, 50, 50))
        expanded_mean[:, :, :] = mean[:, None, None]
        expanded_cov = np.zeros((3, 3, 50, 50))
        expanded_cov[:, :, :, :] = choleskied[:, :, None, None]

        est = mvn.logpdf(mean, expanded_mean + eps, expanded_cov)

        assert np.allclose(true, est)

    def test_Student(self):
        student = cont.Student(3, 1, 2)

        actstuden = stats.t(3, loc=1, scale=2)

        assert student.logpdf(2) == actstuden.logpdf(2)

    def test_Sample(self):
        size = (300, 30, 40)
        uniform = cont.Uniform(-1, 1).sample(size=size)

        assert uniform.values.shape == size and uniform.t_values.min() < -1 and uniform.values.min() > -1

        newvals = stats.uniform.rvs(size=size)

        uniform.values = newvals

        assert np.all(newvals == uniform.values)

        transformed = stats.norm.rvs(size=size)

        uniform.t_values = transformed

        assert np.all(np.abs(uniform.transform(uniform.values) - transformed) < 1e-12)