from .nessmc2 import NESSMC2
from .ness import FixedWidthNESS


class SMC2FW(NESSMC2):
    """
    Implements the SMC2 FW algorithm of Ajay Jasra and Yan Zhou.
    """

    def make_second(self, filter_, particles, **kwargs):
        return FixedWidthNESS(filter_, particles, **kwargs)
