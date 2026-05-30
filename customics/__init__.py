import importlib.metadata

from . import datasets, metrics, tools
from .model import CustOMICS

__version__ = importlib.metadata.version("customics")
