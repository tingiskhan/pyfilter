from typing import Callable
from tqdm import tqdm
from .state import AlgorithmState


class DefaultLogger(object):
    """
    Base class for logging in algorithms.
    """

    def __init__(self, func: Callable[[AlgorithmState], None] = None, log_every_iteration: int = 1):
        """
        Initializes the ``DefaultLogger`` class.

        Args:
            func: Callable that takes as input the current algorithm state.
            log_every_iteration: Integer specifying the frequency at which to call ``func``, calls every
                ``iteration % log_every_iteration``.
        """

        self._func = func or (lambda *args: None)
        self._per_iter = log_every_iteration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
        if exc_type:
            raise exc_val

    def initialize(self, algorithm: "BaseAlgorithm", num_iterations: int) -> "DefaultLogger":
        """
        Initializes the logging class.

        Args:
            algorithm: The algorithm using the logging class.
            num_iterations: The number of iterations to perform.
        """

        return self

    def do_log(self, iteration: int, state: AlgorithmState):
        """
        Performs the actual logging.

        Args:
            iteration: The current iteration of algorithm.
            state: The current state of the algorithm.
        """

        if iteration % self._per_iter == 0:
            self._func(state)

    def teardown(self):
        """
        Tears down the logger class.
        """

        return


class TQDMWrapper(DefaultLogger):
    """
    Logging wrapper for `tqdm`.
    """

    def __init__(self):
        """
        Initializes the ``TQDMWrapper`` class.
        """

        super(TQDMWrapper, self).__init__(func=self.func, log_every_iteration=1)
        self._tqdm_bar = tqdm(total=None)

    def initialize(self, algorithm, num_iterations):
        self._tqdm_bar.total = num_iterations
        self._tqdm_bar.set_description(str(algorithm))

        return self

    def func(self, obj):
        self._tqdm_bar.update(self._per_iter)

    def teardown(self):
        self._tqdm_bar.close()
