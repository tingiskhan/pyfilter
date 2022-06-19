from math import log, exp
from typing import Dict, Tuple, List


class Thresholder(object):
    """
    Object that decides the threshold at which to perform resampling.
    """

    def __init__(self, min_thresh: float, start_thresh: float):
        """
        Initializes the :class:`Thresholder` class.

        Args:
             min_thresh: the minimum allowed threshold.
             start_thresh: the starting threshold.
        """

        self._min = min_thresh
        self._start = start_thresh

    def _mutate_thresh(self, iteration: int, starting_threshold: float) -> float:
        raise NotImplementedError()

    def get_threshold(self, iteration: int) -> float:
        """
        Returns the threshold of the current iteration.

        Args:
            iteration: the current iteration.
        """

        return max(self._mutate_thresh(iteration, self._start), self._min)


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
        Initializes the :class:`DecayingThreshold` class.

        Args:
            half_life: the number of steps at which to achieve a halving of the threshold.
        """

        super().__init__(min_thresh, start_thresh)
        self._alpha = log(2.0) / half_life

    def _mutate_thresh(self, iteration, starting_threshold):
        return exp(-self._alpha * iteration) * starting_threshold


class IntervalThreshold(Thresholder):
    """
    Defines an interval based threshold.
    """

    def _mutate_thresh(self, iteration: int, starting_threshold: float) -> float:
        return next((u[1] for u in self._thresholds if iteration <= u[0]), self._min)

    def __init__(self, thresholds: Dict[int, float], ending_threshold: float):
        """
        Initializes the :class:`IntervalThreshold` class.

        Args:
            thresholds: dictionary specifying the thresholds and the ending numer of num_samples.
            ending_threshold: the ending threshold.
        """

        super(IntervalThreshold, self).__init__(ending_threshold, ending_threshold)
        self._thresholds: List[Tuple[int, float]] = sorted(thresholds.items(), key=lambda u: u[0])
