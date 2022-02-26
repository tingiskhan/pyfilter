from math import log, exp


class Thresholder(object):
    """
    Object that decides the threshold at which to perform resampling.
    """

    def __init__(self, min_thresh: float, start_thresh: float):
        """
        Initializes the ``Thresholder`` class.

        Args:
             min_thresh: The minimum allowed threshold.
             start_thresh: The starting threshold.
        """

        self._min = min_thresh
        self._start = start_thresh

        self._iter = 0

    def _mutate_thresh(self, iteration: int, starting_threshold: float) -> float:
        raise NotImplementedError()

    def get_threshold(self) -> float:
        """
        Returns the threshold of the current iteration.
        """

        self._iter += 1

        return max(self._mutate_thresh(self._iter, self._start), self._min)


class ConstantThreshold(Thresholder):
    """
    Defines a constant threshold.
    """
    
    def __init__(self, threshold: float):
        super(ConstantThreshold, self).__init__(threshold, threshold)

    def _mutate_thresh(self, iteration, starting_threshold):
        return starting_threshold


class DecayingThreshold(Thresholder):
    """
    Defines a decaying threshold.
    """

    def __init__(self, min_thresh: float, start_thresh: float, half_life: int = 1_000):
        """
        Initializes the ``DecayingThreshold`` class.

        Args:
            half_life: The number of steps at which to achieve a halfing of the threshold.
        """

        super().__init__(min_thresh, start_thresh)
        self._alpha = log(2.0) / half_life

    def _mutate_thresh(self, iteration, starting_threshold):
        return exp(-self._alpha * iteration) * starting_threshold
