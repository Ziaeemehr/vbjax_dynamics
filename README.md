# vbjax_dynamics

A JAX-based library for numerical integration of dynamical systems.

**Note:** This package contains code adapted from [vbjax](https://github.com/ins-amu/vbjax) by INS-AMU. The core integration functions in `loops.py` are derived from vbjax's implementation.

## Features

<!-- - **ODE Solvers**: Ordinary Differential Equations
  - Euler method
  - Runge-Kutta methods (RK2, RK4)
  - Adaptive step-size methods

- **DDE Solvers**: Delay Differential Equations
  - Method of steps
  - RK methods with history interpolation

- **SDE Solvers**: Stochastic Differential Equations
  - Euler-Maruyama method
  - Milstein method
  - Runge-Kutta SDE methods

- **SDDE Solvers**: Stochastic Delay Differential Equations
  - Combined approaches for stochastic systems with delays -->

## Installation

```bash
pip install vbjax_dynamics
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic ODE Example

```python
import jax.numpy as jnp
from vbjax_dynamics.ode import RK4Solver

# Define your system
def f(t, y, args):
    return -y

# Solve
solver = RK4Solver(f)
t_span = (0.0, 10.0)
y0 = jnp.array([1.0])
t, y = solver.solve(y0, t_span, dt=0.01)
```

### Configuration

By default, vbjax_dynamics uses JAX's default 32-bit precision. You can configure this:

```python
import vbjax_dynamics

# Option 1: Configure for the entire session
vbjax_dynamics.configure_jax(enable_x64=True)

# Option 2: Use a context manager for temporary changes
with vbjax_dynamics.precision_context(enable_x64=True):
    # High-precision computation here
    result = my_solver.solve(...)

# Option 3: Set it globally yourself before importing
import jax
jax.config.update("jax_enable_x64", True)
import vbjax_dynamics

# Check current configuration
vbjax_dynamics.print_jax_config()
```

## Acknowledgments

This package includes code adapted from [vbjax](https://github.com/ins-amu/vbjax), developed by the Institut de Neurosciences de la Timone (INS-AMU). We are grateful for their work on efficient JAX-based numerical integrators.

## License

MIT License
