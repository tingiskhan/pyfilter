from .affine import AffineProcess
from .state import TimeseriesState


class AffineObservations(AffineProcess):
    def __init__(self, funcs, parameters, increment_dist):
        """
        Class for defining model with affine dynamics in the observable process.
        :param funcs: The functions governing the dynamics of the process
        """

        super().__init__(funcs, parameters, None, increment_dist)

    def sample_path(self, steps, **kwargs):
        raise NotImplementedError("Cannot sample from Observable only!")

    def propagate_state(self, new_values, prev_state):
        return TimeseriesState(prev_state.time_index, new_values)
