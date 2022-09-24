import pytest
import torch

from pyfilter.inference import utils, qmc

BATCH_SHAPES = [torch.Size([]), torch.Size([512, 1]), torch.Size([50, 2, 3])]


@pytest.fixture()
def clear_registry():
    qmc.QuasiRegistry.clear_registry()


class TestUtils(object):
    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_quasi_mv(self, shape, clear_registry):
        mv = utils.QuasiMultivariateNormal(torch.zeros(3), torch.eye(3))
        samples = mv.sample(shape)

        assert (samples.shape == shape + mv.event_shape)
