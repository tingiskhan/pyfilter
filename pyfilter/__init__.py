__version__ = "0.26.0"

from . import filters
from . import inference

from torch.distributions import Distribution

Distribution.set_default_validate_args(False)
