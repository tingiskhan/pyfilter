__version__ = "0.24.8"

from . import filters
from . import inference

from torch.distributions import Distribution

Distribution.set_default_validate_args(False)
