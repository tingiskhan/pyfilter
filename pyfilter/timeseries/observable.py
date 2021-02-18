from .affine import AffineProcess


class AffineObservations(AffineProcess):
    def __init__(self, funcs, parameters, increment_dist):
        """
        Class for defining model with affine dynamics in the observable process.
        :param funcs: The functions governing the dynamics of the process
        """

        super().__init__(funcs, parameters, None, increment_dist)
        self._covariate = None

    def sample_path(self, steps, **kwargs):
        raise NotImplementedError("Cannot sample from Observable only!")
