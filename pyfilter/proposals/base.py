class Proposal(object):
    def __init__(self, model):
        """
        Defines a proposal object for how to draw the particles.
        :param model:
        """

        self._model = model
        self._kernel = None

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