"""
Lung Cancer Detection System - Main Package
"""

from .utils import (
    setup_logging,
    set_seed,
    get_device,
    count_parameters,
    calculate_metrics
)

__version__ = "1.0.0"
__author__ = "AI Medical Research Team"

__all__ = [
    "setup_logging",
    "set_seed",
    "get_device",
    "count_parameters",
    "calculate_metrics"
]
