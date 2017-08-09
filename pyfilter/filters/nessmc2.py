from .smc2 import SMC2
from .ness import NESS
import numpy as np
import pandas as pd


class NESSMC2(SMC2):
    def __init__(self, model, particles, *args, handshake=0.2, **kwargs):
        super().__init__(model, particles, *args, **kwargs)

        self._hs = handshake
        self._switched = False

        self._smc2 = SMC2(model, particles, *args, **kwargs)
        self._ness = NESS(model, particles, *args, **kwargs)

        self._filter = self._ness._filter = self._smc2._filter

    def filter(self, y):
        if self._smc2._ior < self._hs * self._td.shape[0]:
            self._smc2.filter(y)
        else:
            if not self._switched:
                print('\n===== Switching to NESS =====')
                self._ness._filter = self._smc2._filter
                self._switched = True

            self._ness.filter(y)

        return self

    def longfilter(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)

        # ===== SMC2 needs the entire dataset ==== #
        self._td = self._smc2._td = data

        for i in range(data.shape[0]):
            self.filter(data[i])

        self._td = self._smc2._td = None

        return self