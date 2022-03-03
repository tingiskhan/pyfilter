from torch.nn import Module
from .container import BufferTuples


class BaseState(Module):
    """
    Base class for state like objects.
    """

    def __init__(self):
        """
        Initializes the ``BaseState`` class.
        """

        super(BaseState, self).__init__()
        self.tensor_tuples = BufferTuples()
