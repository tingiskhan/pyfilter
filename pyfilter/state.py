from collections import OrderedDict
from typing import Any
from typing import OrderedDict as tOrderedDict

from .container import TensorContainer


class BaseResult(dict):
    """
    Base class for state like objects.
    """

    def __init__(self):
        """
        Internal initializer for :class:`BaseResult`.
        """

        super().__init__()
        self.tensor_tuples = TensorContainer()

    def exchange_tensor_tuples(self, other: "BaseResult"):
        """
        Exchanges the :attr:`tensor_tuples` of self with that of ``other``.

        Args:
            other (BaseResult): the other state to exchange with.
        """

        for k, v in other.tensor_tuples.items():
            self.tensor_tuples[k] = v

    def state_dict(self) -> tOrderedDict[str, Any]:
        """
        Converts ``self`` to a dictionary.
        """

        return OrderedDict({"tensor_tuples": self.tensor_tuples.state_dict()})

    def load_state_dict(self, state_dict: tOrderedDict[str, Any]):
        """
        Loads state from existing state dictionary.

        Args:
            state_dict (OrderedDict[str, Any]): state dictionary to load from.
        """

        self.tensor_tuples.load_state_dict(state_dict["tensor_tuples"])
