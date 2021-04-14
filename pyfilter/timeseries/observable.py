from .affine import AffineProcess


class AffineObservations(AffineProcess):
    def __init__(self, funcs, parameters, increment_dist):
        """
        Class for defining model with affine dynamics in the observable process.

        :param funcs: The functions governing the dynamics of the process
        """

        super().__init__(funcs, parameters, None, increment_dist)

    def initial_sample(self, shape=None):
        raise NotImplementedError("Cannot sample from Observable only!")

    @property
    def n_dim(self) -> int:
        return len(self.increment_dist().event_shape)

    @property
    def num_vars(self) -> int:
        return self.increment_dist().event_shape.numel()

    def sample_path(self, steps, **kwargs):
        raise NotImplementedError("Cannot sample from Observable only!")

    def forward(self, x, time_increment=1.0):
        return super(AffineObservations, self).forward(x, 0.0)

    propagate = forward
