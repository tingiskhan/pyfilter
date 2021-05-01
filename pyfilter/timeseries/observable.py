from functools import lru_cache
from .affine import AffineProcess


class AffineObservations(AffineProcess):
    """
    Class for defining model with affine dynamics in the observable process.
    """

    def __init__(self, funcs, parameters, increment_dist):
        super().__init__(funcs, parameters, None, increment_dist)

    def initial_sample(self, shape=None):
        raise NotImplementedError("Cannot sample from Observable only!")

    @property
    @lru_cache(maxsize=None)
    def n_dim(self) -> int:
        return len(self.increment_dist().event_shape)

    @property
    @lru_cache(maxsize=None)
    def num_vars(self) -> int:
        return self.increment_dist().event_shape.numel()

    def sample_path(self, steps, **kwargs):
        raise NotImplementedError("Cannot sample from Observable only!")

    def forward(self, x, time_increment=1.0):
        return super(AffineObservations, self).forward(x, 0.0)

    propagate = forward

    def propagate_conditional(self, x, u, parameters=None, time_increment=1.0):
        return super(AffineObservations, self).propagate_conditional(x, u, parameters, 0.0)
