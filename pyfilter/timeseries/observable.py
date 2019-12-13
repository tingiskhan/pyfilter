from .affine import AffineProcess


class AffineObservations(AffineProcess):
    def __init__(self, funcs, theta, noise):
        """
        Class for defining model with affine dynamics in the observable process.
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple of callable
        """

        super().__init__(funcs, theta, None, noise)

    def sample_path(self, steps, **kwargs):
        raise NotImplementedError("Cannot sample from Observable only!")

    def _verify_dimensions(self):
        # TODO: Implement this
        return self


