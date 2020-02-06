from .base import SequentialParticleAlgorithm
from .ness import NESS
from .smc2 import SMC2
from tqdm import tqdm
from ..utils import get_ess
import torch
from ..module import TensorContainer


class NESSMC2(SequentialParticleAlgorithm):
    def __init__(self, filter_, particles, switch=500, smc2_th=0.5, update_switch=True, smc2_kernel=None, **nesskwargs):
        """
        Implements a hybrid of the NESS and SMC2 algorithm, as recommended in the NESS article. That is, we use the
        SMC2 algorithm for the first part of the series and then switch to NESS when it becomes too computationally
        demanding to use the SMC2.
        :param switch: At which point to switch inference, in number of observations
        :type switch: int
        :param update_switch: Whether to perform MCMC move on switch if ESS is below threshold of NESS
        :type update_switch: bool
        """

        super().__init__(filter_, particles)

        self._switch = switch
        self._switched = False
        self._updateonhandshake = update_switch

        # ===== Set some key-worded arguments ===== #
        self._smc2 = SMC2(self.filter, particles, threshold=smc2_th, kernel=smc2_kernel)
        self._ness = NESS(self.filter, particles, **nesskwargs)

    @property
    def logged_ess(self):
        return torch.cat((self._smc2.logged_ess, self._ness.logged_ess))

    def initialize(self):
        self._smc2.initialize()
        return self

    def fit(self, y, bar=True):
        self._iterator = self._smc2._iterator = self._ness._iterator = tqdm(y, desc=str(self))

        for yt in self._iterator if bar else y:
            self.update(yt)

        self._iterator = self._smc2._iterator = self._ness._iterator = None

        return self

    def _update(self, y):
        if len(self._smc2._y) < self._switch:
            return self._smc2.update(y)

        if not self._switched:
            self._switched = True

            threshold = self._ness._kernel._th * self._smc2._w_rec.shape[0]
            if get_ess(self._smc2._w_rec) <  threshold and self._updateonhandshake:
                self._smc2.rejuvenate()

            self._ness._w_rec = self._smc2._w_rec
            self._ness._logged_ess = TensorContainer(self._smc2.logged_ess[-1])
            self._iterator.set_description(desc=str(self._ness))

        return self._ness.update(y)

    def predict(self, steps, aggregate=True, **kwargs):
        if not self._switched:
            return self._smc2.predict(steps, aggregate=aggregate, **kwargs)

        return self._ness.predict(steps, aggregate=aggregate, **kwargs)