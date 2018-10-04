from ..timeseries.model import StateSpaceModel


class Proposal(object):
    def __init__(self):
        """
        Defines a proposal object for how to draw the particles.
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

        return self

    def draw(self, y, x, size=None, *args, **kwargs):
        """
        Defines the method for drawing proposals.
        :param y: The current observation
        :type y: np.ndarray|float|torch.Tensor
        :param x: The previous hidden states
        :type x: torch.Tensor
        :param size: The size which to draw
        :param args: Additional arguments
        :param kwargs: Additional kwargs
        :rtype: torch.Tensor
        """

        raise NotImplementedError()

    def weight(self, y, xn, xo, *args, **kwargs):
        """
        Defines the method for weighting observations.
        :param y: The current observation
        :type y: np.ndarray|float|torch.Tensor
        :param xn: The new state
        :type xn: torch.Tensor
        :param xo: The old state
        :type xo: torch.Tensor
        :param args: Additional arguments
        :param kwargs: Additional kwargs
        :rtype: torch.Tensor
        """

        raise NotImplementedError()

    def resample(self, inds):
        """
        For proposals where some of the data is stored locally. As this is only necessary for a few of the proposals,
        it need not be implemented for all.
        :param inds: The indicies to resample
        :type inds: torch.Tensor
        :return: Self
        :rtype: Proposal
        """

        return self