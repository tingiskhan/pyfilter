from typing import Callable
from tqdm import tqdm
from .state import AlgorithmState


class DefaultLogger(object):
    """
    Base class for performing logging in algorithms.
    """

    def __init__(self, func: Callable[[AlgorithmState], None] = None, log_every_iteration: int = 1):
        self._func = func or (lambda *args: None)
        self._per_iter = log_every_iteration

    def initialize(self, algorithm, num_iterations: int):
        return

    def do_log(self, iteration, state):
        if iteration % self._per_iter == 0:
            self._func(state)

    def close(self):
        return


class TQDMWrapper(DefaultLogger):
    """
    Wrapper for `tqdm`.
    """

    def __init__(self):
        super(TQDMWrapper, self).__init__(func=self.func, log_every_iteration=1)
        self._tqdm = tqdm(total=None)

    def initialize(self, algorithm, num_iterations):
        self._tqdm.total = num_iterations
        self._tqdm.set_description(str(algorithm))

    def set_num_iterations(self, iterations):
        self._tqdm.total = iterations

    def func(self, obj):
        self._tqdm.update(self._per_iter)

    def close(self):
        self._tqdm.close()
