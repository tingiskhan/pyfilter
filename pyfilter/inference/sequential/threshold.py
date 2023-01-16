from math import exp, log
from typing import Dict, List, Tuple


class Thresholder(object):
    """
    Object that decides the threshold at which to perform resampling.
    """

    def __init__(self, min_thresh: float, start_thresh: float):
        """
        Internal initializer for :class:`Thresholder`.

        Args:
            min_thresh (float): minimum allowed threshold.
            start_thresh (float): starting threshold.
        """

        self._min = min_thresh
        self._start = start_thresh

    def _mutate_thresh(self, iteration: int, starting_threshold: float) -> float:
        raise NotImplementedError()

    def get_threshold(self, iteration: int) -> float:
        """
        Returns the threshold of the current iteration.

        Args:
            iteration (int): current iteration.
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
        Internal initializer for :class:`DecayingThreshold`.

        Args:
            min_thresh (float): see :class:`Thresholder`.
            start_thresh (float):  see :class:`Thresholder`.
            half_life (int, optional): required number of steps to halve ``start_thresh``. Defaults to 1_000.
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
        Internal initializer for :class:`IntervalThreshold`.

        Args:
            thresholds (Dict[int, float]): dictionary specifying the thresholds and the ending numer of num_samples.
            ending_threshold (float): final threshold.

        Example:
            The following example constructs a :class:`Thresholder` in which we use 50% for the first 100 observations,
            and then use 10% for the remaining.
            >>> from pyfilter.inference.sequential.threshold import IntervalThreshold
            >>>
            >>> threshold = IntervalThreshold({100: 0.5}, 0.1)

        """

        super(IntervalThreshold, self).__init__(ending_threshold, ending_threshold)
        self._thresholds: List[Tuple[int, float]] = sorted(thresholds.items(), key=lambda u: u[0])
