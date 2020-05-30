from .nessmc2 import NESSMC2
from .ness import FixedWidthNESS


class SMC2FW(NESSMC2):
    def __init__(self, filter_, particles, switch=500, smc2_kw=None, ness_kw=None):
        """
        Implements the SMC2 FW algorithm of Ajay Jasra and Yan Zhou.
        """

        super().__init__(filter_, particles, switch, smc2_kw, ness_kw)

    def make_second(self, filter_, particles, **kwargs):
        return FixedWidthNESS(filter_, particles, **kwargs)
