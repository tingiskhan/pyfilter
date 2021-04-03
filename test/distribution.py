import unittest
from pyfilter.distributions import Prior, DistributionWrapper
from torch.distributions import Exponential, StudentT


class MyTestCase(unittest.TestCase):
    def test_PositivePrior(self):
        prior = Prior(Exponential, rate=0.1)
        dist = DistributionWrapper(StudentT, df=prior)

        samples = dist()


if __name__ == '__main__':
    unittest.main()
