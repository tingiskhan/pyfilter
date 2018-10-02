from .apf import APF
from .sisr import SISR
from .rapf import RAPF
from ..proposals.linearized import Linearized as Linz
from .upf import UPF, GlobalUPF
from .ukf import UKF
from .klf import KalmanLaplace


class Linearized(SISR):
    def __init__(self, model, particles, *args, **kwargs):
        super().__init__(model, particles, *args, proposal=Linz(), **kwargs)