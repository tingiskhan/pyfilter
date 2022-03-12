from torch.nn import Module
from .container import BufferIterable


class BaseState(Module):
    """
    Base class for state like objects.
    """

    def __init__(self, maxlen: int = None):
        """
        Initializes the ``BaseState`` class.
        """

        super(BaseState, self).__init__()
        self.tensor_tuples = BufferIterable()
