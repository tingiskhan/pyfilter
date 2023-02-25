import torch
import torch.quasirandom as qr

from ..constants import EPS2


class EngineContainer(object):
    """
    Container for QMC engines.
    """

    def __init__(self, dim: int, randomize: bool):
        """
        Internal initializer for :class:`_EngineContainer`.

        Args:
            dim (int): dimension of sample space.
            randomize (bool): whether to randomize.
        """

        self._engine = qr.SobolEngine(dim, scramble=True)
        self._randomize = randomize
        self._rotation_vector: torch.Tensor = None

    def sample(self, shape: torch.Size) -> torch.Tensor:
        """
        Draws samples from the QMC engine.

        Args:
            shape (torch.Size): shape to draw.
        """

        numel = shape.numel()

        probs = self._engine.draw(numel)

        if shape.numel() == 1:
            probs.squeeze_(0)

        if self._randomize:
            # TODO: Verify below. From the Quasi MH-paper it seems as though the rotation vector should be constant
            #  across samples
            if self._rotation_vector is None:
                self._rotation_vector = torch.empty(probs.shape[-1], device=probs.device).uniform_()

            probs = (probs + self._rotation_vector).remainder(1.0)

        # NB: Same as in nchopin/particles to avoid "degeneracy"
        safe_probs = 0.5 + (1.0 - EPS2) * (probs - 0.5)

        new_shape = shape + torch.Size([self._engine.dimension])
        return safe_probs.reshape(new_shape)
