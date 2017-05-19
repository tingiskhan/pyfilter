from .meta import Base
from pyfilter.distributions.continuous import Distribution


class Observable(Base):
    def __init__(self, funcs, theta, noise):
        """
        Object for defining the observable part of an HMM.
        :param funcs: A tuple of functions
        :type funcs: tuple of functions
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple of functions
        :param theta: The parameters governing the dynamics
        :type theta: tuple
        :param noise: The noise governing the noise process.
        :type noise: Distribution
        """
        super().__init__((None, None), funcs, theta, (None, noise))