import unittest
import torch
from pyfilter.distributions import Prior, DistributionWrapper, JointDistribution
from torch.distributions import Exponential, StudentT, Independent


def make_joint_distribution():
    dist_1 = Exponential(rate=1.0)
    dist_2 = Independent(StudentT(df=torch.ones(2)), 1)

    return JointDistribution(dist_1, dist_2)


class DistributionTests(unittest.TestCase):
    def test_PositivePrior(self):
        prior = Prior(Exponential, rate=0.1)
        dist = DistributionWrapper(StudentT, df=prior)

        samples = dist()

    def test_JointDistributionMasksAndShape(self):
        joint_distribution = make_joint_distribution()

        self.assertEqual(joint_distribution.masks[0], 0)
        self.assertEqual(joint_distribution.masks[1], slice(1, 2))

        self.assertEqual(joint_distribution.event_shape, torch.Size([3]))

    def test_JointDistributionSamples(self):
        joint_distribution = make_joint_distribution()
        samples = joint_distribution.sample()

        self.assertEqual(samples.shape, torch.Size([3]))

        more_sample_shape = torch.Size([1000, 300])
        more_samples = joint_distribution.sample(more_sample_shape)

        self.assertEqual(more_samples.shape, torch.Size([*more_sample_shape, 3]))

    def test_JointDistributionLogProb(self):
        joint_distribution = make_joint_distribution()
        shape = torch.Size([1000, 300])

        samples = joint_distribution.sample(shape)
        log_prob = joint_distribution.log_prob(samples)

        self.assertEqual(log_prob.shape, shape)

    def test_JointDistributionEntropy(self):
        joint_distribution = make_joint_distribution()

        expected = joint_distribution.distributions[0].entropy() + joint_distribution.distributions[1].entropy()
        self.assertEqual(joint_distribution.entropy(), expected)

    def test_CheckGradient(self):
        joint_distribution = make_joint_distribution()

        samples = joint_distribution.sample()

        joint_distribution.distributions[0].rate.requires_grad_(True)
        log_prob = joint_distribution.log_prob(samples)

        self.assertTrue(log_prob.requires_grad)

        log_prob.backward()
        self.assertIsNotNone(joint_distribution.distributions[0].rate.grad)

    def test_CheckExpand(self):
        joint_dist = make_joint_distribution()

        new_shape = torch.Size([1000, 10])
        expanded = joint_dist.expand(new_shape)

        self.assertEqual(expanded.batch_shape, new_shape)


if __name__ == "__main__":
    unittest.main()
