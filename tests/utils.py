import pytest
import torch
from pyfilter.utils import broadcast_all
from torch.distributions import Normal, utils
from pyfilter.distributions import Prior


@pytest.fixture()
def tensors():
    return [torch.empty(100).normal_(), torch.tensor(0.0)]


@pytest.fixture()
def mixed(tensors):
    return tensors + [Prior(Normal, loc=0.0, scale=1.0), torch.tensor(2.0)]


@pytest.fixture()
def only_priors(tensors):
    return [Prior(Normal, loc=i, scale=1.0) for i in range(3)]


class TestUtils(object):
    def test_broadcast_all_tensors(self, tensors):
        torch_broadcast = utils.broadcast_all(*tensors)
        broadcast = broadcast_all(*tensors)

        for correct, value in zip(torch_broadcast, broadcast):
            assert (correct == value).all()

    def test_broadcast_all_mixed(self, mixed):
        torch_broadcast = utils.broadcast_all(*(v for v in mixed if not isinstance(v, Prior)))
        broadcast = broadcast_all(*mixed)

        assert len(torch_broadcast) + 1 == len(broadcast)

        for correct, value in zip(torch_broadcast, (v for v in broadcast if isinstance(v, torch.Tensor))):
            assert (correct == value).all()

        assert mixed[-2] in broadcast

    def test_broadcast_all_same(self, only_priors):
        torch_broadcast = broadcast_all(*only_priors)

        for i, p in enumerate(torch_broadcast):
            assert p.loc == i
