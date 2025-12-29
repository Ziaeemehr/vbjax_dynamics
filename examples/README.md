# vbjax_dynamics Examples and Tutorials

Complete guide for using **vbjax_dynamics** - a high-performance numerical integration library for Ordinary Differential Equations (ODEs), Stochastic Differential Equations (SDEs), and Delay Differential Equations (DDEs) using JAX.

> 📖 **New to vbjax_dynamics?** Start with the [Quick Start](../README.md#quick-start) in the main README, then come back here for detailed tutorials.

## Overview

This directory contains comprehensive examples demonstrating all features of vbjax_dynamics:

- **ODE Integration**: Ordinary differential equations with various methods
- **SDE Integration**: Stochastic differential equations (both manual and automatic noise)
- **DDE Integration**: Delay differential equations
- **Parallel Computing**: Using `jax.vmap` for multiple trajectories
- **Configuration**: Precision and JAX settings

## Example Files in This Directory

- `tutorial.py` - Comprehensive tutorial running all examples with accuracy comparisons
- `introduction_examples.py` - Quick examples with visualizations
- `sde_noise_comparison.py` - Detailed comparison of `make_sde` vs `make_sde_auto`
- `parallel_sde_vmap.py` - Advanced parallel computing with vmap
- `minimal_config_example.py` - Configuration utilities demonstration
- `benchmark_sde_performance.py` - Performance benchmarking (moved from tests)

## Installation

```bash
pip install vbjax_dynamics
```

For development with examples:
```bash
cd vbjax_dynamics
pip install -e ".[dev]"
```

## Quick Start

### Example 1: Simple ODE

```python
import jax.numpy as jnp
from vbjax_dynamics.loops import make_ode

# Define the differential equation: dx/dt = -x
def dfun(x, p):
    return -x

# Setup
dt = 0.01
t_max = 5.0
ts = jnp.arange(0, t_max, dt)
x0 = 1.0

# Create integrator and solve
step, loop = make_ode(dt, dfun, method='rk4')
x = loop(x0, ts, None)

print(f"x(0) = {x[0]:.4f}, x(T) = {x[-1]:.4f}")
```

### Example 2: System of ODEs (Harmonic Oscillator)

```python
def harmonic_oscillator(state, p):
    """state = [position, velocity]"""
    x, v = state
    return jnp.array([v, -x])  # [dx/dt, dv/dt]

x0 = jnp.array([1.0, 0.0])
step, loop = make_ode(dt, harmonic_oscillator, method='rk4')
states = loop(x0, ts, None)
```

### Example 3: Stochastic Differential Equation

**Option 1: Pre-generate noise** (more control)
```python
from jax import random
from vbjax_dynamics.loops import make_sde

def drift(x, p):
    return -p * x  # Drift coefficient

def diffusion(x, p):
    return 0.5  # Diffusion coefficient

# Generate noise beforehand
key = random.PRNGKey(42)
n_steps = int(t_max / dt)
zs = random.normal(key, (n_steps,))

# Solve
theta = 1.0
step, loop = make_sde(dt, drift, diffusion)
x = loop(2.0, zs, theta)
```

**Option 2: Automatic noise** (simpler)
```python
from jax import random
from vbjax_dynamics.loops import make_sde_auto

# Solve with automatic noise generation
step, loop = make_sde_auto(dt, drift, diffusion)
key = random.PRNGKey(42)
n_steps = int(t_max / dt)
x = loop(2.0, n_steps, theta, key)  # Noise generated internally!
```

> 💡 See [SDE_NOISE_OPTIONS.md](SDE_NOISE_OPTIONS.md) for detailed comparison

### Example 4: Delay Differential Equation

```python
from vbjax_dynamics.loops import make_dde

def delayed_system(xt, x, t, p):
    """DDE: dx/dt = -x(t-τ)"""
    tau_steps = p
    x_delayed = xt[t - tau_steps]
    return -x_delayed

# Setup with delay
dt = 0.01
tau = 1.0  # delay time
tau_steps = int(tau / dt)
n_steps = 100

# Create history buffer
buffer_size = tau_steps + 1 + n_steps
history = jnp.ones(buffer_size)
history = history.at[tau_steps + 1:].set(0.0)

# Solve
step, loop = make_dde(dt, tau_steps, delayed_system)
buf_final, x = loop(history, tau_steps, t=0)
```

## API Reference

### `make_ode(dt, dfun, method='heun', adhoc=None)`

Create an ODE integrator.

**Parameters:**
- `dt` (float): Time step size
- `dfun` (callable): Function `dfun(x, p)` that returns dx/dt
- `method` (str): Integration method - `'euler'`, `'heun'`, or `'rk4'`
- `adhoc` (callable, optional): Function `f(x, p)` for post-step corrections

**Returns:**
- `step`: Single step function `step(x, t, p)`
- `loop`: Full integration function `loop(x0, ts, p)`

**Example:**
```python
def dfun(x, p):
    return -p * x  # dx/dt = -p*x

step, loop = make_ode(dt=0.01, dfun=dfun, method='rk4')
x = loop(x0=1.0, ts=jnp.arange(0, 5, 0.01), p=1.0)
```

---

### `make_sde(dt, dfun, gfun, adhoc=None, return_euler=False, unroll=10)`

Create an SDE integrator using the stochastic Heun method.

**Parameters:**
- `dt` (float): Time step size
- `dfun` (callable): Drift function `dfun(x, p)` returning drift coefficient
- `gfun` (callable or float): Diffusion function `gfun(x, p)` or constant diffusion
- `adhoc` (callable, optional): Post-step correction function
- `return_euler` (bool): If True, also return Euler estimates
- `unroll` (int): Loop unroll factor for performance

**Returns:**
- `step`: Single step function `step(x, z_t, p)`
- `loop`: Full integration function `loop(x0, zs, p)` where `zs` are noise samples

**Example:**
```python
def drift(x, p):
    theta = p
    return -theta * x

def diffusion(x, p):
    return 0.5

# Generate noise
zs = random.normal(key, (n_steps,))

step, loop = make_sde(dt=0.01, dfun=drift, gfun=diffusion)
x = loop(x0=1.0, zs=zs, p=1.0)
```

---

### `make_sde_auto(dt, dfun, gfun, adhoc=None, unroll=10)`

Create an SDE integrator with **automatic noise generation** (simplified API).

**Parameters:**
- `dt` (float): Time step size
- `dfun` (callable): Drift function `dfun(x, p)`
- `gfun` (callable or float): Diffusion function or constant
- `adhoc` (callable, optional): Post-step correction function
- `unroll` (int): Loop unroll factor for performance

**Returns:**
- `step`: Single step function `step(x, z_t, p)`
- `loop`: Full integration function `loop(x0, n_steps, p, key)` - noise generated internally!

**Example:**
```python
def drift(x, p):
    return -p * x

# Automatic noise generation - simpler!
step, loop = make_sde_auto(dt=0.01, dfun=drift, gfun=0.5)
key = random.PRNGKey(42)
x = loop(x0=1.0, n_steps=1000, p=1.0, key=key)
```

**Comparison:**
- Use `make_sde` for custom noise or advanced control
- Use `make_sde_auto` for simpler code with Gaussian noise
- See [SDE_NOISE_OPTIONS.md](SDE_NOISE_OPTIONS.md) for details

---

### `make_dde(dt, nh, dfun, unroll=10, adhoc=None)`

Create a delay differential equation integrator.

**Parameters:**
- `dt` (float): Time step size
- `nh` (int): Maximum delay in time steps
- `dfun` (callable): Function `dfun(xt, x, t, p)` where:
  - `xt`: History buffer
  - `x`: Current state
  - `t`: Current time index in buffer
  - `p`: Parameters
- `unroll` (int): Loop unroll factor
- `adhoc` (callable, optional): Post-step correction

**Returns:**
- `step`: Single step function
- `loop`: Full integration function `loop(buffer, p, t=0)`

**Example:**
```python
def dfun(xt, x, t, p):
    tau_steps = p
    return -xt[t - tau_steps]  # dx/dt = -x(t-τ)

tau_steps = 100
buffer_size = tau_steps + 1 + n_steps
history = jnp.ones(buffer_size)

step, loop = make_dde(dt=0.01, nh=tau_steps, dfun=dfun)
buf, x = loop(history, tau_steps)
```

---

### `make_sdde(dt, nh, dfun, gfun, unroll=1, zero_delays=False, adhoc=None)`

Create a stochastic delay differential equation integrator.

**Parameters:**
- `dt` (float): Time step size
- `nh` (int): Maximum delay in time steps
- `dfun` (callable): Drift function `dfun(xt, x, t, p)`
- `gfun` (callable or float): Diffusion function or constant
- `unroll` (int): Loop unroll factor
- `zero_delays` (bool): Include predictor in history (performance vs accuracy trade-off)
- `adhoc` (callable, optional): Post-step correction

**Returns:**
- `step`: Single step function
- `loop`: Full integration function

## Integration Methods

### Accuracy Comparison

For a smooth ODE with step size `dt`:

| Method | Order | Error | Speed | Use Case |
|--------|-------|-------|-------|----------|
| Euler  | 1 | O(dt) | Fastest | Quick prototyping, very smooth problems |
| Heun   | 2 | O(dt²) | Medium | Good balance, default choice |
| RK4    | 4 | O(dt⁴) | Slower | High accuracy requirements |

### When to Use Each Method

**Euler**: 
- Fastest execution
- Use for very smooth problems or when speed is critical
- Good for quick prototyping

**Heun** (default):
- Best balance of speed and accuracy
- Recommended for most applications
- 2nd order accurate

**RK4**:
- Highest accuracy among available methods
- Use when precision is critical
- Required for stiff or complex dynamics

## Comparison with Other Libraries

### vs SciPy

**Advantages of JAX Loops:**
- 10-100x faster with JIT compilation
- GPU/TPU support
- Automatic differentiation through integrators
- Vectorization over multiple initial conditions

**Advantages of SciPy:**
- Adaptive step size methods
- Better for stiff problems
- More mature error control
- Wider method selection (RK45, DOP853, etc.)

**Recommendation**: Use JAX loops for speed-critical applications, parameter fitting (with gradients), or GPU acceleration. Use SciPy for one-off integrations or stiff problems.

### Accuracy Validation

Example comparing RK4 methods on Lorenz system:

```python
# JAX
step, loop = make_ode(0.001, lorenz, method='rk4')
x_jax = loop(x0, ts, params)

# SciPy
sol = solve_ivp(lorenz_scipy, [0, t_max], x0, t_eval=ts, method='RK45')
x_scipy = sol.y.T

# Difference
diff = np.linalg.norm(x_jax - x_scipy, axis=1)
# Typical max difference: ~1e-6 to 1e-8
```

## Running Examples

The repository includes comprehensive examples:

### Full Tutorial
```bash
python tutorial.py
```
Runs all examples with detailed accuracy comparisons against SciPy.

### Quick Examples with Plots
```bash
python examples.py
```
Generates visualizations for:
1. Simple ODE (exponential decay)
2. Harmonic oscillator (energy conservation)
3. Stochastic ODE (Ornstein-Uhlenbeck)
4. Delay differential equation
5. Method comparison

Output: PNG files with plots for each example.

## Performance Tips

1. **Use JIT compilation**: The `loop` functions are pre-compiled with `@jax.jit`
2. **Batch initial conditions**: Use `jax.vmap` to solve multiple initial conditions in parallel
3. **GPU acceleration**: Set `JAX_PLATFORM_NAME=gpu` for GPU execution
4. **Unroll parameter**: Adjust `unroll` in SDEs/DDEs for better performance
5. **Memory**: For long simulations with delays, use `make_continuation` helper

## Common Issues and Solutions

### Issue: "TracerError" or "ConcretizationError"

**Solution**: Ensure all array operations use JAX functions (`jnp` not `np`), and avoid Python control flow. Use `jax.lax.cond` for conditionals.

### Issue: Numerical instability

**Solution**: 
- Reduce `dt` (time step)
- Use higher-order method (RK4 instead of Euler)
- Check problem formulation

### Issue: Slow first run

**Solution**: This is JIT compilation. Subsequent runs will be much faster. Warmup with a dummy run if needed.

### Issue: Memory error with DDEs

**Solution**: 
- Use `make_continuation` for long simulations
- Reduce buffer size if possible
- Process data in chunks

## Advanced Usage

### Automatic Differentiation

```python
def solve_ode(params):
    step, loop = make_ode(dt, lambda x, p: -p * x, method='rk4')
    return loop(x0, ts, params)

# Gradient of final state w.r.t. parameters
grad_fn = jax.grad(lambda p: solve_ode(p)[-1])
gradient = grad_fn(1.0)
```

### Vectorization Over Initial Conditions

```python
# Solve for multiple initial conditions in parallel
x0_batch = jnp.array([1.0, 2.0, 3.0, 4.0])
step, loop = make_ode(dt, dfun, method='rk4')

# Vectorize over first argument (x0)
loop_vmap = jax.vmap(loop, in_axes=(0, None, None))
x_batch = loop_vmap(x0_batch, ts, params)
# Shape: (4, len(ts)) for each initial condition
```

### Custom Post-Step Corrections

```python
def adhoc(x, p):
    """Enforce positivity constraint"""
    return jnp.maximum(x, 0.0)

step, loop = make_ode(dt, dfun, method='rk4', adhoc=adhoc)
```

## Citation

If you use this library in your research, please cite the original vbjax project.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please submit issues or pull requests.

## References

1. Butcher, J. C. (2016). Numerical methods for ordinary differential equations.
2. Kloeden, P. E., & Platen, E. (1992). Numerical solution of stochastic differential equations.
3. Shampine, L. F., & Thompson, S. (2001). Solving DDEs in MATLAB.
