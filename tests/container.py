from pyfilter.container import TensorTuple, BufferDict
from pyfilter.state import StateWithTensorTuples
import pytest
import torch


@pytest.fixture
def tensors():
    return list(torch.empty((50, 10)).normal_())


class TestContainers(object):
    def test_tensor_tuple(self, tensors):
        tt = TensorTuple(*tensors[:-1])

        assert len(tt.tensors) == 49

        tt.append(tensors[-1])

        assert (len(tt.tensors) == 50) and (tt.tensors[-1] is tensors[-1])

    def test_persisting_state_with_tensor(self, tensors):
        state = StateWithTensorTuples()

        key = "temp"
        state.tensor_tuples[key] = TensorTuple(*tensors)

        state_dict = state.state_dict()

        final_key = f"tensor_tuples.{key}"
        assert (final_key in state_dict) and isinstance(state_dict[final_key], tuple)

        new_state = StateWithTensorTuples()
        new_state.load_state_dict(state_dict)

        assert (new_state.tensor_tuples[key].values() == state.tensor_tuples[key].values()).all()

    def test_bufferdict(self):
        buffer_dict = BufferDict(
            {"parameter_0": torch.empty((200,)).normal_()}
        )

        assert (len(buffer_dict.values()) == 1) and ("parameter_0" in buffer_dict)

        buffer_dict["parameter_1"] = torch.tensor(0.0)

        assert (len(buffer_dict.values()) == 2) and ("parameter_1" in buffer_dict)


TestContainers().test_bufferdict()