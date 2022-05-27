import itertools
import random
import numpy as np
import pytest
import torch
import stochproc.timeseries as ts
from pyfilter.filters import particle as part
from pykalman import KalmanFilter

from .models import linear_models


def construct_filters(particles=500, **kwargs):
    particle_types = (part.SISR, part.APF)

    for pt in particle_types:
        yield lambda m: pt(m, particles, proposal=part.proposals.Bootstrap(), **kwargs)
        yield lambda m: pt(m, particles, proposal=part.proposals.Linearized(n_steps=5), **kwargs)
        yield lambda m: pt(m, particles, proposal=part.proposals.Linearized(n_steps=5, use_second_order=True), **kwargs)


def add_nan(y, number_of_nans):
    rand_ints = torch.randint(0, y.shape[0], size=(number_of_nans,))

    if y.dim() == 1:
        y[rand_ints] = float("nan")
    else:
        rand_col = random.randint(0, y.shape[-1] - 1)
        y[rand_ints, rand_col] = float("nan")

    return y


class TestFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    PARALLEL_FILTERS = 20
    SERIES_LENGTH = 100
    PREDICTION_STEPS = 5
    STATE_RECORD_LENGTH = 5
    NUMBER_OF_NANS = 10

    def _compare_kalman_mean(self, kalman_mean, means):
        assert ((means - kalman_mean) / kalman_mean).abs().median() < self.RELATIVE_TOLERANCE

    @pytest.mark.parametrize("models, filter_", itertools.product(linear_models(), construct_filters()))
    def test_compare_with_kalman_filter(self, models, filter_):
        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)

        kalman_mean, _ = kalman_model.filter(y)
        kalman_ll = kalman_model.loglikelihood(y)

        f = filter_(model)
        result = f.batch_filter(torch.from_numpy(y).float())

        assert len(result.states) == 1
        assert ((result.loglikelihood - kalman_ll) / kalman_ll).abs() < self.RELATIVE_TOLERANCE

        means = result.filter_means[1:]
        if model.hidden.n_dim < 1:
            means.unsqueeze_(-1)

        self._compare_kalman_mean(kalman_mean, means)
