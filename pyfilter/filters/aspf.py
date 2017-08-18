from .base import BaseFilter


def _propagate_params(params):
    """
    Propagates the parameters.
    :param params:
    :return:
    """


class ASPF(BaseFilter):
    def filter(self, y):
        self._model.p_apply()