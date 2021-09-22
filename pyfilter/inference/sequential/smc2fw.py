from .nessmc2 import NESSMC2
from .ness import FixedWidthNESS


class SMC2FW(NESSMC2):
    """
    Implements the SMC2 FW algorithm of Ajay Jasra and Yan Zhou, which amounts to running the ``SMC2`` algorithm on some
    subset of the data, and then switch and use ``FixedWidthNESS`` for the remainder.
    """

    def make_second(self, filter_, particles, **kwargs):
        return FixedWidthNESS(filter_, particles, **kwargs)
