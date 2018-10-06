from .smc2 import SMC2
from .ness import NESS


class NESSMC2(SMC2):
    def __init__(self, filter_, particles, handshake=200, threshold=0.5, nesskwargs=None):
        """
        Implements a hybrid of the NESS and SMC2 algorithm, as recommended in the NESS article. That is, we use the
        SMC2 algorithm for the first part of the series and then switch to NESS when it becomes too computationally
        demanding to use the SMC2.
        :param handshake: At which point to switch algorithms, in number of observations
        :type handshake: int
        """
        super().__init__(filter_, particles)

        self._hs = handshake
        self._switched = False

        # ===== Set some key-worded arguments ===== #
        nk = nesskwargs or {}
        self._smc2 = SMC2(self._filter, particles, threshold=threshold)
        self._ness = NESS(self._filter, particles, shrinkage=nk.pop('shrinkage', 0.95), p=nk.pop('p', 1),  **nk)

    def initialize(self):
        self._smc2.initialize()
        return self

    def update(self, y):
        if len(self._smc2._td) < self._hs:
            return self._smc2.update(y)

        if not self._switched:
            self._switched = True
            self._filter = self._ness._filter = self._smc2._filter

        return self._ness.update(y)