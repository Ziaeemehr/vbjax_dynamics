# Acknowledgments

## vbjax

This package contains code adapted from **vbjax**, developed by the Institut de Neurosciences de la Timone (INS-AMU).

- **Repository**: https://github.com/ins-amu/vbjax
- **Authors**: INS-AMU team
- **License**: Apache-2.0

### Adapted Code

The following files/functions contain code derived from vbjax:

- `src/vbjax_dynamics/loops.py`: Core integration functions including:
  - `make_ode()` - ODE integrator builder
  - `make_sde()` - SDE integrator builder with explicit noise
  - `make_dde()` - DDE integrator builder
  - `make_sdde()` - SDDE integrator builder
  - `make_continuation()` - Continuation method builder

**Note**: `make_sde_auto()` is a new addition to this package, not derived from vbjax.

### Modifications

We have made the following modifications to the original vbjax code:

1. **Removed global configuration**: The original code set `jax_enable_x64=True` globally. We removed this to give users control over precision settings.

2. **Added safety warnings**: Added warning system for unsafe default random key usage to help users avoid common pitfalls with JAX's random number generation.

3. **Enhanced documentation**: Added comprehensive docstrings and examples.

### New Additions (Not from vbjax)

The following components are original contributions to this package:

1. **`utils.py` module**: Configuration utilities including `configure_jax()`, `precision_context()`, `print_jax_config()`, and `get_default_dtype()` to give users easy control over JAX settings.

2. **`make_sde_auto()` function**: A convenient wrapper around `make_sde()` that automatically handles random key splitting for easier use.

3. **Comprehensive test suite**: Pure pytest-based tests for accuracy, JIT compilation, and vmap functionality.

4. **Reorganized structure**: Adapted the package structure to follow modern Python packaging best practices with a src-layout.

## Gratitude

We are deeply grateful to the vbjax developers for their excellent work on efficient JAX-based numerical integrators. Their implementation provided a solid foundation for this package.

If you use this package in academic work, please consider citing both this package and the original vbjax project.

---

*If you believe you should be credited here, please open an issue or pull request.*
