"""
Heston Stochastic Volatility Model with Particle Filtering.

This package provides a complete implementation of the Heston model
as a state-space model with particle filtering capabilities.
"""

from .model import HestonModel, HestonSSM
from .estimation import diagnose_pmmh_mixing

__all__ = ['HestonModel', 'HestonSSM', 'diagnose_pmmh_mixing']
__version__ = '0.1.0'

