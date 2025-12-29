# vbjax_dynamics

A JAX-based library for numerical integration of dynamical systems.

**Note:** This package contains code adapted from [vbjax](https://github.com/ins-amu/vbjax) by INS-AMU. The core integration functions in `loops.py` are derived from vbjax's implementation.

## Features

- **ODE Integration**: Ordinary Differential Equations
  - Efficient loop-based integrators with JIT compilation
  - Full support for `jax.vmap` for parallel trajectory computation
  
- **SDE Integration**: Stochastic Differential Equations  
  - `make_sde()`: Integration with pre-generated noise arrays
  - `make_sde_auto()`: Automatic noise generation from random keys
  - Euler-Maruyama scheme
  - Fully reproducible with random seeds

- **DDE Integration**: Delay Differential Equations
  - Support for fixed delays
  - History function interpolation

- **SDDE Integration**: Stochastic Delay Differential Equations
  - Combined stochastic and delay dynamics

- **Continuation Methods**: Parameter continuation for bifurcation analysis

- **Configuration Utilities**: Easy control over JAX settings
  - `configure_jax()`: Global configuration
  - `precision_context()`: Temporary precision changes
  - `print_jax_config()`: Diagnostic information

- **JAX-Native**: 
  - JIT compilation for speed
  - Automatic differentiation ready
  - GPU/TPU compatible
  - Pure functional approach

## Installation

```bash
pip install vbjax_dynamics
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from jax import random, vmap
from vbjax_dynamics.loops import make_sde_auto

# Define Ornstein-Uhlenbeck process
def drift(x, p):
    return -p[0] * x  # -theta * x

def diffusion(x, p):
    return p[1]  # sigma

# Create integrator
dt = 0.01
step, loop = make_sde_auto(dt, drift, diffusion)

# Single trajectory
x0 = 2.0
params = (1.0, 0.5)  # (theta, sigma)
n_steps = 1000
key = random.PRNGKey(42)

trajectory = loop(x0, n_steps, params, key)
print(f"Final value: {trajectory[-1]:.4f}")

# Multiple trajectories in parallel with vmap
n_traj = 100
keys = random.split(key, n_traj)
trajectories = vmap(lambda k: loop(x0, n_steps, params, k))(keys)
print(f"Mean: {jnp.mean(trajectories[:, -1]):.4f}")
```

For more examples, see the [`examples/`](examples/) directory.

## Documentation

- **[Examples and Tutorials](examples/README.md)**: Complete guide with detailed examples for ODE, SDE, DDE, and SDDE integration
- **[Testing Guide](tests/README.md)**: Information about running and writing tests
- **[Acknowledgments](ACKNOWLEDGMENTS.md)**: Credits and attribution

## Acknowledgments

This package includes code adapted from [vbjax](https://github.com/ins-amu/vbjax), developed by the Institut de Neurosciences de la Timone (INS-AMU). We are grateful for their work on efficient JAX-based numerical integrators.

## License

MIT License
