import pytest
from pyfilter.timeseries import models
import torch


@pytest.fixture
def timeseries_models():
    return (
        models.AR(0.0, 0.99, 0.05),
        models.LocalLinearTrend(torch.tensor([1e-3, 1e-2])),
        models.OrnsteinUhlenbeck(0.01, 0.0, 0.05),
        models.OrnsteinUhlenbeck(0.01 * torch.ones(2), torch.zeros(2), 0.05 * torch.ones(2)),
        models.RandomWalk(0.05),
        models.RandomWalk(0.05 * torch.ones(2)),
        models.Verhulst(0.01, 1.0, 0.05, 1.0),
        models.SemiLocalLinearTrend(0.0, 0.99, torch.tensor([1e-3, 1e-2])),
        models.UCSV(0.01, torch.tensor([0.0, 1.0])),
    )


class TestTimeseries(object):
    timesteps = 1000

    def test_sample_path(self, timeseries_models):
        for m in timeseries_models:
            path = m.sample_path(self.timesteps)

            assert path.shape[0] == self.timesteps

    def test_sample_path_batched(self, timeseries_models):
        samples = torch.Size([10, 10])
        for m in timeseries_models:
            path = m.sample_path(self.timesteps, samples=samples)

            assert path.shape[1:3] == samples

    def test_sample_num_steps(self, timeseries_models):
        num_steps = 5

        for m in timeseries_models:
            try:
                m.num_steps = num_steps

                x = m.initial_sample()
                x = m.propagate(x)

                assert x.time_index == num_steps
            finally:
                m.num_steps = 1
