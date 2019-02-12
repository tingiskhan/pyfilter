from .base import SequentialAlgorithm
from .ness import NESS
from .smc2 import SMC2
from tqdm import tqdm
from ..resampling import systematic
from ..utils import get_ess
import torch


class NESSMC2(SequentialAlgorithm):
    def __init__(self, filter_, particles, handshake=500, smc2_threshold=0.5, resampling=systematic,
                 update_on_handshake=False, **nesskwargs):
        """
        Implements a hybrid of the NESS and SMC2 algorithm, as recommended in the NESS article. That is, we use the
        SMC2 algorithm for the first part of the series and then switch to NESS when it becomes too computationally
        demanding to use the SMC2.
        :param handshake: At which point to switch algorithms, in number of observations
        :type handshake: int
        :param update_on_handshake: Whether to perform MCMC move on handshake if ESS is below threshold of NESS
        :type update_on_handshake: bool
        """

        super().__init__(filter_)

        self._hs = handshake
        self._switched = False
        self._updateonhandshake = update_on_handshake

        # ===== Set some key-worded arguments ===== #
        self._smc2 = SMC2(self.filter, particles, resampling=resampling, threshold=smc2_threshold)
        self._ness = NESS(self._smc2.filter, particles, resampling=resampling, **nesskwargs)

    @property
    def logged_ess(self):
        return torch.stack(self._smc2._logged_ess + self._ness._logged_ess, dim=0)

    def initialize(self):
        self._smc2.initialize()
        return self

    def fit(self, y):
        self._iterator = self._smc2._iterator = self._ness._iterator = tqdm(y, desc=str(self))

        for yt in self._iterator:
            self.update(yt)

        return self

    def update(self, y):
        if len(self._smc2._y) < self._hs:
            return self._smc2.update(y)

        if not self._switched:
            self._switched = True

            if get_ess(self._smc2._w_rec) < self._ness._th * self._smc2._w_rec.shape[0] and self._updateonhandshake:
                self._smc2._rejuvenate()

            self._ness._w_rec = self._smc2._w_rec
            self.filter = self._ness.filter = self._smc2.filter

            self._iterator.set_description(desc=str(self._ness))

        return self._ness.update(y)