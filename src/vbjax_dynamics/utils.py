"""
Utility functions for vbjax_dynamics
"""

import jax
from contextlib import contextmanager
from typing import Optional


def configure_jax(
    enable_x64: bool = False,
    device: Optional[str] = None,
    platform: Optional[str] = None,
):
    """
    Configure JAX settings for the entire session.
    
    This is a convenience function that users can call at the start of their
    scripts to set up JAX according to their needs.
    
    Parameters
    ----------
    enable_x64 : bool, default False
        Enable 64-bit precision. If False, uses 32-bit (default JAX behavior).
        64-bit gives higher accuracy but uses more memory and may be slower.
    device : str, optional
        Specify device (e.g., 'cpu', 'gpu', 'tpu')
    platform : str, optional
        Specify platform (e.g., 'cpu', 'gpu', 'tpu')
        
    Example
    -------
    >>> from vbjax_dynamics.utils import configure_jax
    >>> 
    >>> # Enable 64-bit precision for high accuracy
    >>> configure_jax(enable_x64=True)
    >>> 
    >>> # Force CPU execution
    >>> configure_jax(enable_x64=True, platform='cpu')
    
    Notes
    -----
    This function sets global JAX configuration that affects all subsequent
    JAX operations. Call it at the beginning of your script before any
    JAX computations.
    """
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
        print("JAX: 64-bit precision enabled")
    
    if platform is not None:
        jax.config.update("jax_platform_name", platform)
        print(f"JAX: Platform set to {platform}")
    
    # Print current configuration
    print(f"JAX: Using {jax.default_backend()} backend")


@contextmanager
def precision_context(enable_x64: bool = True):
    """
    Context manager for temporary precision changes.
    
    Use this when you need 64-bit precision for only part of your computation.
    The precision setting is restored to its original value after the context exits.
    
    Parameters
    ----------
    enable_x64 : bool, default True
        Enable 64-bit precision within the context
        
    Yields
    ------
    None
    
    Example
    -------
    >>> from vbjax_dynamics.utils import precision_context
    >>> import jax.numpy as jnp
    >>> 
    >>> # Normal 32-bit computation
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> print(x.dtype)  # float32
    >>> 
    >>> # Temporary 64-bit computation
    >>> with precision_context(enable_x64=True):
    >>>     y = jnp.array([1.0, 2.0, 3.0])
    >>>     print(y.dtype)  # float64
    >>> 
    >>> # Back to 32-bit
    >>> z = jnp.array([1.0, 2.0, 3.0])
    >>> print(z.dtype)  # float32
    """
    # Save original state
    original_x64 = jax.config.read("jax_enable_x64")
    
    try:
        # Set new state
        jax.config.update("jax_enable_x64", enable_x64)
        yield
    finally:
        # Restore original state
        jax.config.update("jax_enable_x64", original_x64)


def get_default_dtype():
    """
    Get the current default dtype based on JAX configuration.
    
    Returns
    -------
    dtype
        jnp.float64 if x64 is enabled, otherwise jnp.float32
        
    Example
    -------
    >>> from vbjax_dynamics.utils import get_default_dtype
    >>> import jax.numpy as jnp
    >>> 
    >>> dtype = get_default_dtype()
    >>> x = jnp.zeros(10, dtype=dtype)
    """
    import jax.numpy as jnp
    if jax.config.read("jax_enable_x64"):
        return jnp.float64
    else:
        return jnp.float32


def print_jax_config():
    """
    Print current JAX configuration settings.
    
    Useful for debugging and understanding the current environment.
    
    Example
    -------
    >>> from vbjax_dynamics.utils import print_jax_config
    >>> print_jax_config()
    JAX Configuration:
    ------------------
    Backend: gpu
    64-bit precision: False
    Devices: [gpu(id=0)]
    """
    print("JAX Configuration:")
    print("-" * 40)
    print(f"Backend: {jax.default_backend()}")
    print(f"64-bit precision: {jax.config.read('jax_enable_x64')}")
    print(f"Devices: {jax.devices()}")
    print(f"Default dtype: {get_default_dtype()}")


# Re-export commonly used functions for convenience
__all__ = [
    "configure_jax",
    "precision_context",
    "get_default_dtype",
    "print_jax_config",
]
