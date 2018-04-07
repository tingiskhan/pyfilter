from ..timeseries.model import StateSpaceModel
from ..utils.stategradient import StateGradient, NumericalStateGradient


class Proposal(object):
    def __init__(self, *args, **kwargs):
        """
        Defines a proposal object for how to draw the particles.
        :param args: Any arguments passed to the proposal
        :param kwargs: Any kwargs passed to the proposal
        """

        self._model = None
        self._kernel = None
        self._nested = None

        self._meaner = lambda x: x
        self._sg = None

    def set_model(self, model, nested=False):
        """
        Sets the model and all required attributes.
        :param model: The model to ues
        :type model: StateSpaceModel
        :param nested: A boolean for specifying if the algorithm is running nested PFs
        :type nested: bool
        :return: Self
        :rtype: Proposal
        """

        self._model = model
        self._nested = nested
        self._sg = NumericalStateGradient(self._model)

        return self

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

    def resample(self, inds):
        """
        For proposals where some of the data is stored locally. As this is only necessary for a few of the proposals,
        it need not be implemented for all.
        :param inds: The indicies to resample.
        :return:
        """

        return self