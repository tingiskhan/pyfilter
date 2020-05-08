from .base import SequentialParticleAlgorithm
from .ness import NESS
from .smc2 import SMC2
from tqdm import tqdm
import torch
from ..module import TensorContainer


class NESSMC2(SequentialParticleAlgorithm):
    def __init__(self, filter_, particles, switch=500, update_switch=True, smc2kw=None, nkw=None):
        """
        Implements a hybrid of the NESS and SMC2 algorithm, as recommended in the NESS article. That is, we use the
        SMC2 algorithm for the first part of the series and then switch to NESS when it becomes too computationally
        demanding to use the SMC2.
        :param switch: At which point to switch inference, in number of observations
        :type switch: int
        :param update_switch: Whether to perform MCMC move on switch if ESS is below threshold of NESS
        :type update_switch: bool
        :param smc2kw: Any key worded arguments to SMC2
        :type smc2kw: dict[str, object]
        :param nkw: Any key worded arguments for NESS
        :type nkw: dict[str, object]
        """

        super().__init__(filter_, particles)

        self._switch = switch
        self._switched = False
        self._updateonhandshake = update_switch

        # ===== Set some key-worded arguments ===== #
        self._smc2 = SMC2(self.filter, particles, **(smc2kw or dict()))
        self._ness = NESS(self.filter, particles, **(nkw or dict()))

    @property
    def logged_ess(self):
        return torch.cat((self._smc2.logged_ess, self._ness.logged_ess))

    def modules(self):
        return {
            '_smc2': self._smc2,
            '_ness': self._ness
        }

    def initialize(self):
        self._smc2.initialize()
        return self

    @property
    def _w_rec(self):
        if self._switched:
            return self._ness._w_rec

        return self._smc2._w_rec

    @_w_rec.setter
    def _w_rec(self, x):
        return

    def _fit(self, y, bar=True):
        iterator = y
        if bar:
            self._iterator = self._smc2._iterator = self._ness._iterator = iterator = tqdm(y, desc=str(self))

        for yt in iterator:
            self.update(yt)

        self._iterator = self._smc2._iterator = self._ness._iterator = None

        return self

    def _update(self, y):
        if len(self._smc2._y) < self._switch:
            return self._smc2.update(y)

        if not self._switched:
            self._switched = True

            if self._smc2._logged_ess[-1] < self._ness._threshold and self._updateonhandshake:
                self._smc2.rejuvenate()
            else:
                self._ness._logged_ess = TensorContainer(self._smc2._logged_ess[-1])

            self._ness._w_rec = self._smc2._w_rec
            self._iterator.set_description(desc=str(self._ness))

        return self._ness.update(y)

    def predict(self, steps, aggregate=True, **kwargs):
        if not self._switched:
            return self._smc2.predict(steps, aggregate=aggregate, **kwargs)

        return self._ness.predict(steps, aggregate=aggregate, **kwargs)