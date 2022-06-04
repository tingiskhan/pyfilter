import itertools
import pytest
import torch
from pyfilter.filters import particle as part
import numpy as np

from .models import linear_models


def construct_filters(particles=1_000, **kwargs):
    particle_types = (part.SISR, part.APF)

    for pt in particle_types:
        yield lambda m: pt(m, particles, proposal=part.proposals.Bootstrap(), **kwargs)
        yield lambda m: pt(m, particles, proposal=part.proposals.Linearized(n_steps=5), **kwargs)
        yield lambda m: pt(m, particles, proposal=part.proposals.Linearized(n_steps=5, use_second_order=True), **kwargs)

        linear_proposal = part.proposals.LinearGaussianObservations(parameter_index=0)
        yield lambda m: pt(m, particles, proposal=linear_proposal)


BATCH_SIZES = [
    torch.Size([]),
    torch.Size([10]),
]

PARAMETERS = itertools.product(linear_models(), construct_filters(), BATCH_SIZES)


class TestParticleFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    SERIES_LENGTH = 100

    @pytest.mark.parametrize("models, filter_, batch_size", PARAMETERS)
    def test_compare_with_kalman_filter(self, models, filter_, batch_size):
        np.random.seed(123)

        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)

        kalman_mean, _ = kalman_model.filter(y)

        if len(batch_size) > 0:
            kalman_mean = kalman_mean[:, None]

        kalman_ll = kalman_model.loglikelihood(y)

        f: part.ParticleFilter = filter_(model)
        f.set_batch_shape(batch_size)
        result = f.batch_filter(torch.from_numpy(y).float())

        assert len(result.states) == 1
        assert (((result.loglikelihood - kalman_ll) / kalman_ll).abs() < self.RELATIVE_TOLERANCE).all()

        means = result.filter_means[1:]
        std = result.filter_variance[1:].sqrt()

        low = means - std
        high = means + std

        if model.hidden.n_dim < 1:
            low.unsqueeze_(-1)
            high.unsqueeze_(-1)

        low = low.numpy()
        high = high.numpy()

        assert ((low <= kalman_mean) & (kalman_mean <= high)).all()