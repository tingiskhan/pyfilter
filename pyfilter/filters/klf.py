from .ukf import UKF
from ..utils.unscentedtransform import _covcalc
from scipy.optimize import minimize


class KalmanLaplace(UKF):
    _initialized = False

    def initialize(self):
        self._initialize_parameters()
        self._ut.initialize(self.ssm.hidden.i_mean())

        return self

    def filter(self, y):
        if not self._initialized:
            func = lambda x: -(self.ssm.weight(y, x) + self.ssm.hidden.i_weight(x))
            start = self.ssm.hidden.i_mean()
            self._initialized = True
        else:
            func = lambda x: -(self.ssm.weight(y, x) + self.ssm.hidden.weight(x, self._old_x))
            start = self.ssm.propagate_apf(self._old_x)

        minimzed = minimize(func, start)

        return self