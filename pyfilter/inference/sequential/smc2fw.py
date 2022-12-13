from .ness import FixedWidthNESS
from .nessmc2 import NESSMC2


class SMC2FW(NESSMC2):
    """
    Implements the `SMC2 FW`_ algorithm of Ajay Jasra and Yan Zhou, which amounts to running the
    :class:`pyfilter.inference.sequential.SMC2` algorithm on some subset of the data, and then switch and use
    :class:`FixedWidthNESS` for the remainder.

    .. _`SMC2 FW`: https://arxiv.org/pdf/1503.00266.pdf
    """

    def make_second(self, filter_, particles, **kwargs):
        return FixedWidthNESS(filter_, particles, **kwargs)
