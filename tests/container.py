from pyfilter.container import BufferIterable, BufferDict
from pyfilter.state import BaseState
import pytest
import torch


@pytest.fixture
def tensors():
    return list(torch.empty((50, 10)).normal_())


class TestContainers(object):
    def test_tensor_tuple(self, tensors):
        buffer_tuples = BufferIterable()
        buffer_tuples["temp"] = tensors

        assert isinstance(buffer_tuples["temp"], list)
        state_dict = buffer_tuples.state_dict()

        new_tuples = BufferIterable()
        new_tuples.load_state_dict(state_dict)

        for v1, v2 in zip(buffer_tuples["temp"], new_tuples["temp"]):
            assert (v1 == v2).all()

        assert len(buffer_tuples["temp"]) == len(new_tuples["temp"])

    def test_serialize_state(self, tensors):
        state = BaseState()
        state.tensor_tuples["temp"] = tensors

        state_dict = state.state_dict()

        new_state = BaseState()
        new_state.load_state_dict(state_dict)

        for v1, v2 in zip(state.tensor_tuples["temp"], new_state.tensor_tuples["temp"]):
            assert (v1 == v2).all()

        assert len(state.tensor_tuples["temp"]) == len(new_state.tensor_tuples["temp"])

    def test_serialize_empty(self):
        state = BaseState()
        state.tensor_tuples["temp"] = tuple()

        state_dict = state.state_dict()

        new_state = BaseState()
        new_state.load_state_dict(state_dict)

    def test_apply(self, tensors):
        buffer_tuples = BufferIterable()
        buffer_tuples["temp"] = tensors

        if not torch.cuda.is_available():
            return

        buffer_tuples = buffer_tuples.cuda()

        for k in buffer_tuples.keys():
            assert buffer_tuples.get_as_tensor(k).device.type == "cuda"

    def test_bufferdict(self):
        buffer_dict = BufferDict(
            {"parameter_0": torch.empty((200,)).normal_()}
        )

        assert (len(buffer_dict.values()) == 1) and ("parameter_0" in buffer_dict)

        buffer_dict["parameter_1"] = torch.tensor(0.0)

        assert (len(buffer_dict.values()) == 2) and ("parameter_1" in buffer_dict)
