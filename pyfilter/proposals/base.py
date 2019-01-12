from ..timeseries.model import StateSpaceModel


class Proposal(object):
    def __init__(self):
        """
        Defines a proposal object for how to draw the particles.
        """
        self._model = None
        self._kernel = None

    def set_model(self, model):
        """
        Sets the model and all required attributes.
        :param model: The model to ues
        :type model: StateSpaceModel
        :return: Self
        :rtype: Proposal
        """

        self._model = model

        return self

    def draw(self, y, x):
        """
        Defines the method for drawing proposals.
        :param y: The current observation
        :type y: np.ndarray|float|torch.Tensor
        :param x: The previous hidden states
        :type x: torch.Tensor
        :rtype: torch.Tensor
        """

        raise NotImplementedError()

    def weight(self, y, xn, xo):
        """
        Defines the method for weighting observations.
        :param y: The current observation
        :type y: np.ndarray|float|torch.Tensor
        :param xn: The new state
        :type xn: torch.Tensor
        :param xo: The old state
        :type xo: torch.Tensor
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