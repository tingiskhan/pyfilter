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
        Exchanges the :prop:`.tensor_tuples` of self with that of ``other``.

        Args:
            other: the other state to exchange with.
        """

        for k, v in other.tensor_tuples.items():
            self.tensor_tuples[k] = v

        return
