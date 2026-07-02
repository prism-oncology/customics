import importlib.metadata
import logging

import mudata
from ._logging import configure_logger
from . import datasets, metrics
from .utils import get_shared_samples, get_sub_mudata, split_mudata, toy_dataset, prepare_input
from .visualization import plot_cohort_overview
from .model import CustOMICS

mudata.set_options(pull_on_update=False)

__version__ = importlib.metadata.version("customics")

log = logging.getLogger("customics")
configure_logger(log)
