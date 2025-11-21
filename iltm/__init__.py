from importlib.metadata import version

from .inference_interface import iLTMRegressor, iLTMClassifier
from .hyperparameter_search_space import (
    get_hyperparameter_search_space,
    sample_hyperparameters,
    AVAILABLE_CHECKPOINTS,
)

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    'iLTMRegressor',
    'iLTMClassifier',
    'get_hyperparameter_search_space',
    'sample_hyperparameters',
    'AVAILABLE_CHECKPOINTS',
    '__version__',
]