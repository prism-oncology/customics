import importlib.metadata
import logging

from ._logging import configure_logger
from . import datasets, metrics
from .utils import get_shared_samples, get_sub_mudata, toy_dataset, prepare_input
from .model import CustOMICS

__version__ = importlib.metadata.version("customics")

log = logging.getLogger("customics")
configure_logger(log)
