from typing import Callable
from torch import Tensor
from tqdm import tqdm


class LoggingWrapper(object):
    def __init__(self, func: Callable[[object, int, Tensor], None], per_iter: int):
        """
        Logging wrapper for performing logging at `per_iter`
        :param func: The function to call
        """

        self._func = func
        self._per_iter = per_iter

    def set_num_iter(self, iters):
        return self

    def do_log(self, iteration, model, y):
        if iteration % self._per_iter == 0:
            self._func(model, iteration, y)


class DefaultLogger(LoggingWrapper):
    def __init__(self):
        """
        Default wrapper that does not print anything
        """
        super(DefaultLogger, self).__init__(lambda *u: None, 0)

    def do_log(self, iteration, model, y):
        return


class TqdmWrapper(LoggingWrapper):
    def __init__(self, max_iter: int = None):
        """
        Wrapper for tqdm that prints and displays a status bar
        """
        super(TqdmWrapper, self).__init__(self.func, 1)
        self._tqdm = tqdm(total=max_iter)
        self._initialized = False

    def set_num_iter(self, iters):
        self._tqdm.total = iters

    def func(self, obj, it, y):
        if not self._initialized:
            self._tqdm.set_description(str(obj))
            self._initialized = True

        self._tqdm.update(self._per_iter)

