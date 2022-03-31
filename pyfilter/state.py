from torch.nn import Module
from .container import BufferIterable


class BaseState(Module):
    """
    Base class for state like objects.
    """

    def __init__(self):
        """
        Initializes the ``BaseState`` class.
        """

        super(BaseState, self).__init__()
        self.tensor_tuples = BufferIterable()

    def exchange_tensor_tuples(self, other: "BaseState"):
        """
        Exchanges the ``.tensor_tuples`` of self with that of other.

        Args:
            other: The other state to exchange with.
        """

        for k, v in other.tensor_tuples.items():
            self.tensor_tuples[k] = v

        return
