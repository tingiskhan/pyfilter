import pytest
import torch
from pyfilter.distributions import Prior, DistributionWrapper, JointDistribution
from torch.distributions import (
    Exponential,
    StudentT,
    Independent,
    AffineTransform,
    TransformedDistribution,
    ExpTransform,
    Normal,
)


@pytest.fixture
def joint_distribution():
    dist_1 = Exponential(rate=1.0)
    dist_2 = Independent(StudentT(df=torch.ones(2)), 1)

    return JointDistribution(dist_1, dist_2)


@pytest.fixture
def dist_with_prior():
    prior = Prior(Exponential, rate=0.1)
    return DistributionWrapper(StudentT, df=prior)


class TestDistributions(object):
    def test_prior(self, dist_with_prior):
        dist = dist_with_prior()

        assert (
                isinstance(dist, StudentT) and
                (dist.df > 0).all() and
                (dist.df == dist_with_prior().df) and
                len(tuple(dist_with_prior.priors())) == 1
        )

    def test_prior_sample_parameter(self, dist_with_prior):
        size = torch.Size([1000])

        for parameter, prior in dist_with_prior.parameters_and_priors():
            parameter.sample_(prior, size)

        dist = dist_with_prior()

        assert dist.df.shape == size

    def test_distribution_wrapper(self):
        wrapper = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        dist = wrapper.build_distribution()

        assert (dist.mean == torch.tensor(0.0)) and (dist.variance == torch.tensor(1.0))

    def test_distribution_bad_parameter(self):
        wrapper = DistributionWrapper(Normal, loc=0.0, scale=-1.0, validate_args=True)

        with pytest.raises(ValueError):
            dist = wrapper.build_distribution()


class TestJointDistribution(object):
    def test_mask(self, joint_distribution):
        assert joint_distribution.indices[0] == 0
        assert joint_distribution.indices[1] == slice(1, 3)

        assert joint_distribution.event_shape == torch.Size([3])

    def test_samples(self, joint_distribution):
        samples = joint_distribution.sample()

        assert samples.shape == torch.Size([3])

        more_sample_shape = torch.Size([1000, 300])
        more_samples = joint_distribution.sample(more_sample_shape)

        assert more_samples.shape == torch.Size([*more_sample_shape, 3])

    def test_log_prob(self, joint_distribution):
        shape = torch.Size([1000, 300])

        samples = joint_distribution.sample(shape)
        log_prob = joint_distribution.log_prob(samples)

        assert log_prob.shape == shape

    def test_entropy(self, joint_distribution):
        expected = joint_distribution.distributions[0].entropy() + joint_distribution.distributions[1].entropy()
        assert joint_distribution.entropy() == expected

    def test_gradient(self, joint_distribution):
        samples = joint_distribution.sample()

        joint_distribution.distributions[0].rate.requires_grad_(True)
        log_prob = joint_distribution.log_prob(samples)

        assert log_prob.requires_grad

        log_prob.backward()
        assert joint_distribution.distributions[0].rate.grad is not None

    def test_expand(self, joint_distribution):
        new_shape = torch.Size([1000, 10])
        expanded = joint_distribution.expand(new_shape)

        assert expanded.batch_shape == new_shape

    def test_transform(self, joint_distribution):
        transformed = TransformedDistribution(joint_distribution, AffineTransform(0.0, 0.0))
        samples = transformed.sample()

        assert (samples == 0.0).all()

        mean = 1.0
        transformed = TransformedDistribution(joint_distribution, AffineTransform(mean, 1.0))
        samples = transformed.sample()

        assert transformed.log_prob(samples) == joint_distribution.log_prob(samples - mean)

        exp_transform = TransformedDistribution(joint_distribution, ExpTransform())
        samples = exp_transform.sample((1000,))

        assert (samples >= 0.0).all()
