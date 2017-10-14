from ..timeseries.model import StateSpaceModel
from ..utils.stategradient import StateGradient


class Proposal(object):
    def __init__(self, model, nested=False):
        """
        Defines a proposal object for how to draw the particles.
        :param model: The model to ues
        :type model: StateSpaceModel
        :param nested: A boolean for specifying if the algorithm is running nested PFs
        :type nested: bool
        """

        self._model = model
        self._kernel = None
        self._nested = nested

        if self._nested:
            self._meaner = lambda x: x.mean(axis=-1)[..., None]
        else:
            self._meaner = lambda x: x

        self._sg = StateGradient(self._model)

    def draw(self, y, x, size=None, *args, **kwargs):
        """
        Defines the method for drawing proposals.
        :param y: The current observation
        :param x: The previous hidden states
        :param size: The size which to draw
        :param args: Additional arguments
        :param kwargs: Additional kwargs
        :return:
        """

        raise NotImplementedError()

    def weight(self, y, xn, xo, *args, **kwargs):
        """
        Defines the method for weighting observations.
        :param y: The current observation
        :param xn: The new state
        :param xo: The old state
        :param args: Additional arguments
        :param kwargs: Additional kwargs
        :return:
        """

        raise NotImplementedError()