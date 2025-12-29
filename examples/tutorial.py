"""
Tutorial: Numerical Integration with JAX Loops
==============================================

This tutorial demonstrates how to use the loops.py library for numerical 
integration of ODEs, SDEs, and DDEs using JAX. We'll compare results with 
SciPy and other standard Python libraries for accuracy verification.

Author: Tutorial
Date: December 2025
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from vbjax_dynamics.loops import make_ode, make_sde, make_dde

# Set up JAX configuration
jax.config.update("jax_enable_x64", True)

print("=" * 70)
print("JAX LOOPS INTEGRATION TUTORIAL")
print("=" * 70)


# =============================================================================
# EXAMPLE 1: Simple ODE Integration - Exponential Decay
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 1: ODE Integration - Exponential Decay")
print("=" * 70)
print("\nProblem: dx/dt = -x, x(0) = 1")
print("Analytical solution: x(t) = exp(-t)\n")


def exponential_decay(x, p):
    """Drift function: dx/dt = -x"""
    return -x


# Time parameters
dt = 0.01
t_max = 5.0
ts = jnp.arange(0, t_max, dt)
x0 = 1.0

# Method 1: JAX Loops with different schemes
print("Testing JAX Loops with different integration schemes:")
print("-" * 50)

methods = ['euler', 'heun', 'rk4']
jax_results = {}

for method in methods:
    step, loop = make_ode(dt, exponential_decay, method=method)
    x_jax = loop(x0, ts, None)
    jax_results[method] = x_jax
    
    # Calculate error at final time
    analytical = jnp.exp(-t_max)
    error = jnp.abs(x_jax[-1] - analytical)
    print(f"{method.upper():6s}: x(T) = {x_jax[-1]:.10f}, error = {error:.2e}")

# Method 2: SciPy for comparison
print("\nComparing with SciPy:")
print("-" * 50)

def scipy_decay(x, t):
    return -x

# Using scipy.integrate.odeint
x_scipy_odeint = odeint(scipy_decay, x0, np.array(ts))
error_scipy = np.abs(x_scipy_odeint[-1, 0] - np.exp(-t_max))
print(f"SciPy odeint: x(T) = {x_scipy_odeint[-1, 0]:.10f}, error = {error_scipy:.2e}")

# Using scipy.integrate.solve_ivp (RK45)
sol_ivp = solve_ivp(lambda t, x: scipy_decay(x, t), [0, t_max], [x0], 
                    t_eval=np.array(ts), method='RK45')
x_scipy_ivp = sol_ivp.y[0]
error_ivp = np.abs(x_scipy_ivp[-1] - np.exp(-t_max))
print(f"SciPy RK45:   x(T) = {x_scipy_ivp[-1]:.10f}, error = {error_ivp:.2e}")

# Analytical solution
x_analytical = np.exp(-np.array(ts))
print(f"Analytical:   x(T) = {x_analytical[-1]:.10f}, error = 0.00e+00")


# =============================================================================
# EXAMPLE 2: Harmonic Oscillator (System of ODEs)
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: System of ODEs - Harmonic Oscillator")
print("=" * 70)
print("\nProblem: d²x/dt² = -ω²x, with ω = 1")
print("Rewritten as: dx/dt = v, dv/dt = -ω²x")
print("Initial conditions: x(0) = 1, v(0) = 0")
print("Analytical solution: x(t) = cos(ωt), v(t) = -ω*sin(ωt)\n")


def harmonic_oscillator(state, p):
    """
    state = [x, v]
    p = [omega]
    Returns [dx/dt, dv/dt] = [v, -omega^2 * x]
    """
    x, v = state
    omega = p if p is not None else 1.0
    return jnp.array([v, -omega**2 * x])


# Parameters
dt = 0.01
t_max = 10.0
ts = jnp.arange(0, t_max, dt)
x0 = jnp.array([1.0, 0.0])  # [position, velocity]
omega = 1.0

# JAX Loops integration
step_rk4, loop_rk4 = make_ode(dt, harmonic_oscillator, method='rk4')
states_jax = loop_rk4(x0, ts, omega)

# Analytical solution
x_analytical = np.cos(omega * np.array(ts))
v_analytical = -omega * np.sin(omega * np.array(ts))

# Calculate errors
x_error = np.abs(states_jax[:, 0] - x_analytical)
v_error = np.abs(states_jax[:, 1] - v_analytical)

print(f"JAX RK4 Method:")
print(f"  Position error at t={t_max}: {x_error[-1]:.2e}")
print(f"  Velocity error at t={t_max}: {v_error[-1]:.2e}")
print(f"  Max position error: {np.max(x_error):.2e}")
print(f"  Max velocity error: {np.max(v_error):.2e}")

# Energy conservation check (H = 0.5 * v^2 + 0.5 * omega^2 * x^2)
energy_jax = 0.5 * states_jax[:, 1]**2 + 0.5 * omega**2 * states_jax[:, 0]**2
energy_drift = np.abs(energy_jax[-1] - energy_jax[0])
print(f"  Energy drift: {energy_drift:.2e}")


# =============================================================================
# EXAMPLE 3: Stochastic Differential Equation (SDE)
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Stochastic Differential Equation (SDE)")
print("=" * 70)
print("\nOrnstein-Uhlenbeck Process:")
print("  dx = -θ(x - μ)dt + σ dW")
print("  Parameters: θ = 1.0, μ = 0.0, σ = 0.5")
print("  Initial condition: x(0) = 2.0\n")


def ou_drift(x, p):
    """Drift: -θ(x - μ)"""
    theta, mu, sigma = p
    return -theta * (x - mu)


def ou_diffusion(x, p):
    """Diffusion: σ"""
    theta, mu, sigma = p
    return sigma


# Parameters
dt = 0.01
t_max = 10.0
n_steps = int(t_max / dt)
ts = jnp.arange(0, t_max, dt)
x0 = 2.0
params = (1.0, 0.0, 0.5)  # (theta, mu, sigma)

# Generate random noise for SDE
key = random.PRNGKey(42)
zs = random.normal(key, (n_steps,)) * jnp.sqrt(dt)

# JAX SDE integration
step_sde, loop_sde = make_sde(dt, ou_drift, ou_diffusion)
x_sde = loop_sde(x0, zs, params)

print(f"Initial value: x(0) = {x0}")
print(f"Final value:   x(T) = {x_sde[-1]:.6f}")
print(f"Mean (should approach μ=0): {jnp.mean(x_sde[500:]):.6f}")
print(f"Std (theory: σ/√(2θ)={params[2]/np.sqrt(2*params[0]):.6f}): {jnp.std(x_sde[500:]):.6f}")

# Multiple trajectories for statistical validation
print("\nRunning 1000 trajectories for statistical validation...")
n_traj = 1000
keys = random.split(key, n_traj)
final_values = []

for i in range(n_traj):
    zs_i = random.normal(keys[i], (n_steps,)) * jnp.sqrt(dt)
    x_i = loop_sde(x0, zs_i, params)
    final_values.append(x_i[-1])

final_values = jnp.array(final_values)
mean_final = jnp.mean(final_values)
std_final = jnp.std(final_values)
theoretical_std = params[2] / jnp.sqrt(2 * params[0])

print(f"  Sample mean:      {mean_final:.6f} (theory: {params[1]:.6f})")
print(f"  Sample std:       {std_final:.6f} (theory: {theoretical_std:.6f})")
print(f"  Relative error:   {abs(std_final - theoretical_std)/theoretical_std * 100:.2f}%")


# =============================================================================
# EXAMPLE 4: Delay Differential Equation (DDE)
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Delay Differential Equation (DDE)")
print("=" * 70)
print("\nProblem: dx/dt = -x(t-τ), with delay τ = 1.0")
print("Initial history: x(t) = 1 for t ∈ [-τ, 0]\n")


def delayed_system(xt, x, t, p):
    """
    xt: history buffer
    x: current state
    t: current time index
    p: parameters
    """
    tau_steps = p  # delay in time steps
    x_delayed = xt[t - tau_steps]
    return -x_delayed


# Parameters
dt = 0.01
tau = 1.0  # delay time
tau_steps = int(tau / dt)  # delay in time steps
t_max = 10.0
n_steps = int(t_max / dt)

# Create history buffer
# Buffer needs to hold: [history] + [current] + [future samples]
buffer_size = tau_steps + 1 + n_steps
history = jnp.ones(buffer_size)  # Initial condition: x = 1 for all history
history = history.at[tau_steps + 1:].set(0.0)  # Future samples set to 0 (DDE, no noise)

# JAX DDE integration
step_dde, loop_dde = make_dde(dt, tau_steps, delayed_system)
buf_final, x_dde = loop_dde(history, tau_steps, t=0)

print(f"Initial value: x(0) = {history[tau_steps]:.6f}")
print(f"Final value:   x(T) = {x_dde[-1]:.6f}")
print(f"Solution oscillates and decays as expected")
print(f"First 10 values: {x_dde[:10]}")


# =============================================================================
# EXAMPLE 5: Lorenz System - Chaos and Accuracy
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 5: Lorenz System - Chaotic Dynamics")
print("=" * 70)
print("\nLorenz equations:")
print("  dx/dt = σ(y - x)")
print("  dy/dt = x(ρ - z) - y")
print("  dz/dt = xy - βz")
print("  Parameters: σ=10, ρ=28, β=8/3\n")


def lorenz(state, p):
    """Lorenz system"""
    x, y, z = state
    sigma, rho, beta = p
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    return jnp.array([dx, dy, dz])


# Parameters
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
params_lorenz = (sigma, rho, beta)
x0_lorenz = jnp.array([1.0, 1.0, 1.0])

dt = 0.001
t_max = 10.0
ts_lorenz = jnp.arange(0, t_max, dt)

# JAX integration with RK4
step_lorenz, loop_lorenz = make_ode(dt, lorenz, method='rk4')
states_lorenz = loop_lorenz(x0_lorenz, ts_lorenz, params_lorenz)

print(f"Integrated {len(ts_lorenz)} steps")
print(f"Initial state: ({x0_lorenz[0]:.2f}, {x0_lorenz[1]:.2f}, {x0_lorenz[2]:.2f})")
print(f"Final state:   ({states_lorenz[-1, 0]:.2f}, {states_lorenz[-1, 1]:.2f}, {states_lorenz[-1, 2]:.2f})")

# Compare with SciPy
def lorenz_scipy(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

sol_lorenz = solve_ivp(lorenz_scipy, [0, t_max], np.array(x0_lorenz), 
                       t_eval=np.array(ts_lorenz), method='RK45')

# Calculate difference
diff = np.linalg.norm(states_lorenz - sol_lorenz.y.T, axis=1)
print(f"\nComparison with SciPy RK45:")
print(f"  Mean difference: {np.mean(diff):.6e}")
print(f"  Max difference:  {np.max(diff):.6e}")
print(f"  Final difference: {diff[-1]:.6e}")


# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON: JAX vs NumPy/SciPy")
print("=" * 70)

import time

# Test problem: Lorenz system
n_runs = 100
dt_perf = 0.01
t_max_perf = 5.0
ts_perf = jnp.arange(0, t_max_perf, dt_perf)

# JAX (with JIT compilation)
step_jax, loop_jax = make_ode(dt_perf, lorenz, method='rk4')

# Warmup
_ = loop_jax(x0_lorenz, ts_perf, params_lorenz)

start = time.time()
for _ in range(n_runs):
    _ = loop_jax(x0_lorenz, ts_perf, params_lorenz)
jax_time = (time.time() - start) / n_runs

print(f"\nJAX (JIT-compiled): {jax_time*1000:.3f} ms per run")

# SciPy
start = time.time()
for _ in range(n_runs):
    _ = solve_ivp(lorenz_scipy, [0, t_max_perf], np.array(x0_lorenz), 
                  t_eval=np.array(ts_perf), method='RK45')
scipy_time = (time.time() - start) / n_runs

print(f"SciPy RK45:         {scipy_time*1000:.3f} ms per run")
print(f"Speedup:            {scipy_time/jax_time:.2f}x")


# =============================================================================
# SUMMARY AND BEST PRACTICES
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY AND BEST PRACTICES")
print("=" * 70)
print("""
1. CHOOSING INTEGRATION METHODS:
   - Euler: Fastest but least accurate (O(dt))
   - Heun: Good balance of speed and accuracy (O(dt²))
   - RK4: Most accurate for smooth problems (O(dt⁴))

2. ACCURACY TIPS:
   - Use smaller dt for better accuracy
   - RK4 is recommended for ODEs requiring high accuracy
   - For stiff problems, consider adaptive methods (SciPy)
   - Always validate with known analytical solutions when possible

3. JAX ADVANTAGES:
   - JIT compilation provides significant speedups
   - Automatic differentiation through integrators
   - GPU/TPU acceleration for large problems
   - Vectorization over multiple initial conditions

4. WHEN TO USE EACH TOOL:
   - JAX loops: Fast, parallelizable, differentiable
   - SciPy: Adaptive methods, stiff problems, mature ecosystem
   - Hand-coded: Educational purposes, full control

5. WORKING WITH SDEs:
   - Generate noise externally and pass to integrator
   - Use sufficient trajectories for statistical validation
   - Check convergence properties (weak vs strong)

6. WORKING WITH DDEs:
   - Carefully set up history buffer size
   - Buffer = history + current + future_steps
   - Consider zero_delays parameter for accuracy vs performance
""")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)
