from random import random
import pytest
import torch

from pyfilter.inference import utils, qmc

BATCH_SHAPES = [torch.Size([]), torch.Size([512, 1]), torch.Size([50, 2, 3])]


class TestUtils(object):
    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    @pytest.mark.parametrize("randomize", [True, False])
    def test_quasi_mv(self, shape, randomize):
        dim = 3
        engine = qmc.EngineContainer(dim, randomize=randomize)
        mv = utils.QuasiMultivariateNormal(engine, torch.zeros(dim), torch.eye(dim))
        samples = mv.sample(shape)

        assert (samples.shape == shape + mv.event_shape)
