from .apf import APF
from .sisr import SISR
from .ness import NESS
from .rapf import RAPF
from .smc2 import SMC2
from .nessmc2 import NESSMC2
from ..proposals.linearized import Linearized as Linz
from .upf import UPF, GlobalUPF
from .ukf import UKF


class Linearized(SISR):
    def __init__(self, model, particles, *args, **kwargs):
        super().__init__(model, particles, *args, proposal=Linz, **kwargs)