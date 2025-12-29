"""
vbjax_dynamics: JAX-based integrators for dynamical systems

This package contains code adapted from vbjax (https://github.com/ins-amu/vbjax)
developed by Institut de Neurosciences de la Timone (INS-AMU).
"""

__version__ = "0.1.0"

# from . import ode
# from . import dde
# from . import sde
# from . import sdde
# from . import base
from . import utils
from . import loops

# Import commonly used utilities for convenience
from .utils import configure_jax, precision_context, print_jax_config

__all__ = [
    # "ode",
    # "dde", 
    # "sde",
    # "sdde",
    # "base",
    # "utils",
    "loops",
    "configure_jax",
    "precision_context",
    "print_jax_config",
]
