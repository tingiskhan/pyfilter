from collections import OrderedDict
from typing import Dict, Any

from stochproc.container import BufferIterable


class BaseResult(dict):
    """
    Base class for state like objects.
    """

    def __init__(self):
        """
        Initializes the :class:`BaseResult` class.
        """

        super(BaseResult, self).__init__()
        self.tensor_tuples = BufferIterable()

    def exchange_tensor_tuples(self, other: "BaseResult"):
        """
        Exchanges the :attr:`tensor_tuples` of self with that of ``other``.

        Args:
            other: the other state to exchange with.
        """

        for k, v in other.tensor_tuples.items():
            self.tensor_tuples[k] = v

        return

    def state_dict(self) -> OrderedDict[str, Any]:
        """
        Converts ``self`` to a dictionary.
        """

        return OrderedDict({"tensor_tuples": self.tensor_tuples.state_dict()})

    def load_state_dict(self, state_dict: OrderedDict[str, Any]):
        """
        Loads state from existing state dictionary.

        Args:
            state_dict: state dictionary to load from.
        """

        self.tensor_tuples.load_state_dict(state_dict["tensor_tuples"])

        return
