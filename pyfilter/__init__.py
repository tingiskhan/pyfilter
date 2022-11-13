__version__ = "0.25.1"

from . import filters
from . import inference

from torch.distributions import Distribution

Distribution.set_default_validate_args(False)
