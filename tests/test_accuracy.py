"""
Accuracy Validation Tests
==========================

This script runs validation tests comparing JAX loops with analytical solutions
and SciPy to verify accuracy.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from scipy.integrate import solve_ivp
from vbjax_dynamics.loops import make_ode, make_sde

jax.config.update("jax_enable_x64", True)

def test_exponential_decay():
    """Test ODE: dx/dt = -x with analytical solution x(t) = exp(-t)"""
    
    def dfun(x, p):
        return -x
    
    dt = 0.01
    t_max = 5.0
    ts = jnp.arange(0, t_max, dt)
    x0 = 1.0
    
    # Test all methods
    methods = ['euler', 'heun', 'rk4']
    analytical = jnp.exp(-t_max)
    thresholds = {'euler': 1e-3, 'heun': 1e-5, 'rk4': 1e-10}
    
    for method in methods:
        step, loop = make_ode(dt, dfun, method=method)
        x = loop(x0, ts, None)
        error = abs(x[-1] - analytical)
        
        # Assert the error is within threshold
        assert error < thresholds[method], \
            f"{method} method error {error:.2e} exceeds threshold {thresholds[method]:.2e}"



def test_harmonic_oscillator():
    """Test harmonic oscillator: d²x/dt² = -x with energy conservation check"""
    
    def harmonic(state, p):
        x, v = state
        return jnp.array([v, -x])
    
    dt = 0.01
    t_max = 10.0
    ts = jnp.arange(0, t_max, dt)
    x0 = jnp.array([1.0, 0.0])
    
    step, loop = make_ode(dt, harmonic, method='rk4')
    states = loop(x0, ts, None)
    
    # Check energy conservation
    energy = 0.5 * (states[:, 0]**2 + states[:, 1]**2)
    energy_drift = abs(energy[-1] - energy[0])
    
    # Check against analytical solution
    x_analytical = jnp.cos(t_max)
    v_analytical = -jnp.sin(t_max)
    
    x_error = abs(states[-1, 0] - x_analytical)
    v_error = abs(states[-1, 1] - v_analytical)
    
    # Assert all conditions
    assert energy_drift < 1e-10, f"Energy drift {energy_drift:.2e} exceeds threshold 1e-10"
    assert x_error < 1e-2, f"Position error {x_error:.2e} exceeds threshold 1e-2"
    assert v_error < 1e-2, f"Velocity error {v_error:.2e} exceeds threshold 1e-2"



def test_scipy_comparison():
    """Compare JAX implementation with SciPy on a nonlinear ODE"""
    
    def dfun_jax(state, p):
        x, y = state
        return jnp.array([y, -0.1*y - jnp.sin(x)])
    
    def dfun_scipy(t, state):
        x, y = state
        return [y, -0.1*y - np.sin(x)]
    
    dt = 0.01
    t_max = 10.0
    ts = jnp.arange(0, t_max, dt)
    x0 = jnp.array([1.0, 0.0])
    
    # JAX solution
    step, loop = make_ode(dt, dfun_jax, method='rk4')
    states_jax = loop(x0, ts, None)
    
    # SciPy solution
    sol = solve_ivp(dfun_scipy, [0, t_max], np.array(x0), 
                    t_eval=np.array(ts), method='RK45')
    states_scipy = sol.y.T
    
    # Compare
    diff = np.linalg.norm(states_jax - states_scipy, axis=1)
    max_diff = np.max(diff)
    
    # Note: SciPy uses adaptive RK45, we use fixed-step RK4
    # Differences up to 1e-2 are acceptable for different methods
    assert max_diff < 1e-2, \
        f"Maximum difference {max_diff:.2e} exceeds threshold 1e-2 (JAX vs SciPy)"



def test_stochastic_moments():
    """Test SDE statistical properties using Ornstein-Uhlenbeck process"""
    
    def drift(x, p):
        theta, sigma = p
        return -theta * x
    
    def diffusion(x, p):
        theta, sigma = p
        return sigma
    
    dt = 0.01
    t_max = 10.0
    n_steps = int(t_max / dt)
    theta = 1.0
    sigma = 0.5
    params = (theta, sigma)
    x0 = 2.0
    
    # Run many trajectories
    n_traj = 1000
    key = random.PRNGKey(42)
    keys = random.split(key, n_traj)
    
    step, loop = make_sde(dt, drift, diffusion)
    
    final_values = []
    for k in keys:
        zs = random.normal(k, (n_steps,))
        x = loop(x0, zs, params)
        final_values.append(x[-1])
    
    final_values = jnp.array(final_values)
    
    # Theoretical stationary distribution: N(0, σ²/(2θ))
    mean_theory = 0.0
    std_theory = sigma / jnp.sqrt(2 * theta)
    
    mean_empirical = jnp.mean(final_values)
    std_empirical = jnp.std(final_values)
    
    mean_error = abs(mean_empirical - mean_theory)
    std_error = abs(std_empirical - std_theory) / std_theory
    
    # Generous thresholds for statistical tests
    assert mean_error < 0.05, \
        f"Mean error {mean_error:.4f} exceeds threshold 0.05"
    assert std_error < 0.15, \
        f"Std relative error {std_error:.2%} exceeds threshold 15%"



def test_convergence_order():
    """Test that RK4 has 4th order convergence"""
    
    def dfun(x, p):
        return -x
    
    t_max = 1.0
    dts = [0.1, 0.05, 0.025, 0.0125]
    x0 = 1.0
    analytical = jnp.exp(-t_max)
    
    errors = []
    for dt in dts:
        ts = jnp.arange(0, t_max, dt)
        step, loop = make_ode(dt, dfun, method='rk4')
        x = loop(x0, ts, None)
        error = abs(x[-1] - analytical)
        errors.append(error)
    
    # For 4th order method, halving dt should reduce error by ~16x
    avg_ratio = np.mean([errors[i]/errors[i+1] for i in range(len(errors)-1)])
    expected_ratio = 16.0  # 2^4 for 4th order
    
    # Should be close to 16 (within 20% tolerance)
    tolerance = 0.2 * expected_ratio
    assert abs(avg_ratio - expected_ratio) < tolerance, \
        f"Convergence ratio {avg_ratio:.2f} differs from expected {expected_ratio:.2f} by more than {tolerance:.2f}"


