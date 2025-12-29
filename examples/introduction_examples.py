"""
Quick Start Examples for JAX Loops Integration Library
=======================================================

Simple, runnable examples demonstrating ODE, SDE, and DDE integration.
"""

import os
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from vbjax_dynamics.loops import make_ode, make_sde, make_dde

jax.config.update("jax_enable_x64", True)
path = "outputs"
os.makedirs(path, exist_ok=True)

# =============================================================================
# EXAMPLE 1: Basic ODE - Exponential Decay
# =============================================================================
def example_1_simple_ode():
    """Solve dx/dt = -x with x(0) = 1"""
    print("\n=== Example 1: Simple ODE (Exponential Decay) ===")
    
    # Define the differential equation
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
    
    # Compare with analytical solution
    x_exact = jnp.exp(-ts)
    error = jnp.max(jnp.abs(x - x_exact))
    
    print(f"Final value: {x[-1]:.6f}")
    print(f"Exact value: {x_exact[-1]:.6f}")
    print(f"Max error: {error:.2e}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ts, x, label='Numerical (RK4)')
    plt.plot(ts, x_exact, '--', label='Analytical')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.legend()
    plt.title('Exponential Decay: dx/dt = -x')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(ts, jnp.abs(x - x_exact))
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.title('Integration Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+'/example1_ode.png', dpi=150)
    print(f"Plot saved as '{path}/example1_ode.png'")
    plt.close()
    
    return x


# =============================================================================
# EXAMPLE 2: System of ODEs - Harmonic Oscillator
# =============================================================================
def example_2_harmonic_oscillator():
    """Solve the harmonic oscillator: d²x/dt² = -x"""
    print("\n=== Example 2: Harmonic Oscillator ===")
    
    def dfun(state, p):
        """state = [x, v], returns [dx/dt, dv/dt] = [v, -x]"""
        x, v = state
        return jnp.array([v, -x])
    
    # Setup
    dt = 0.01
    t_max = 20.0
    ts = jnp.arange(0, t_max, dt)
    x0 = jnp.array([1.0, 0.0])  # [position, velocity]
    
    # Solve
    step, loop = make_ode(dt, dfun, method='rk4')
    states = loop(x0, ts, None)
    
    # Analytical solution
    x_exact = jnp.cos(ts)
    v_exact = -jnp.sin(ts)
    
    # Energy (should be conserved)
    energy = 0.5 * (states[:, 0]**2 + states[:, 1]**2)
    
    print(f"Initial energy: {energy[0]:.6f}")
    print(f"Final energy: {energy[-1]:.6f}")
    print(f"Energy drift: {abs(energy[-1] - energy[0]):.2e}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(ts, states[:, 0], label='Numerical')
    plt.plot(ts, x_exact, '--', label='Analytical')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.title('Position vs Time')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(states[:, 0], states[:, 1], label='Numerical')
    plt.plot(x_exact, v_exact, '--', label='Analytical')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend()
    plt.title('Phase Space')
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 3, 3)
    plt.plot(ts, energy)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Total Energy (Should be Constant)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path+'/example2_harmonic.png', dpi=150)
    print(f"Plot saved as '{path}/example2_harmonic.png'")
    plt.close()
    
    return states


# =============================================================================
# EXAMPLE 3: Stochastic ODE - Ornstein-Uhlenbeck Process
# =============================================================================
def example_3_stochastic():
    """Solve the Ornstein-Uhlenbeck process: dx = -θx dt + σ dW"""
    print("\n=== Example 3: Stochastic Differential Equation ===")
    
    def dfun(x, p):
        """Drift: -θx"""
        theta, sigma = p
        return -theta * x
    
    def gfun(x, p):
        """Diffusion: σ"""
        theta, sigma = p
        return sigma
    
    # Parameters
    dt = 0.01
    t_max = 10.0
    theta = 1.0
    sigma = 0.5
    params = (theta, sigma)
    
    n_steps = int(t_max / dt)
    ts = jnp.arange(0, t_max, dt)
    x0 = 2.0
    
    # Generate noise
    key = random.PRNGKey(42)
    zs = random.normal(key, (n_steps,))
    
    # Solve
    step, loop = make_sde(dt, dfun, gfun)
    x = loop(x0, zs, params)
    
    # Generate multiple trajectories
    n_traj = 100
    keys = random.split(key, n_traj)
    trajectories = []
    
    for k in keys:
        zs_i = random.normal(k, (n_steps,))
        x_i = loop(x0, zs_i, params)
        trajectories.append(x_i)
    
    trajectories = jnp.array(trajectories)
    mean_traj = jnp.mean(trajectories, axis=0)
    std_traj = jnp.std(trajectories, axis=0)
    
    # Theoretical values
    mean_theory = x0 * jnp.exp(-theta * ts)
    
    print(f"Initial value: {x0:.2f}")
    print(f"Final mean (empirical): {mean_traj[-1]:.4f}")
    print(f"Final mean (theory): {mean_theory[-1]:.4f}")
    print(f"Stationary std (theory): {sigma/jnp.sqrt(2*theta):.4f}")
    print(f"Stationary std (empirical): {jnp.std(trajectories[:, -1]):.4f}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(ts, x, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Single Trajectory')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    for i in range(min(20, n_traj)):
        plt.plot(ts, trajectories[i], alpha=0.3, color='blue')
    plt.plot(ts, mean_traj, 'r-', linewidth=2, label='Empirical mean')
    plt.plot(ts, mean_theory, 'k--', linewidth=2, label='Theoretical mean')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Multiple Trajectories')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.hist(trajectories[:, -1], bins=30, density=True, alpha=0.7, 
             label='Empirical')
    plt.xlabel('x(T)')
    plt.ylabel('Density')
    plt.title(f'Distribution at t={t_max}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path+'/example3_sde.png', dpi=150)
    print(f"Plot saved as '{path}/example3_sde.png'")
    plt.close()
    
    return trajectories


# =============================================================================
# EXAMPLE 4: Delay Differential Equation
# =============================================================================
def example_4_delay():
    """Solve dx/dt = -x(t-τ) with delay τ"""
    print("\n=== Example 4: Delay Differential Equation ===")
    
    def dfun(xt, x, t, p):
        """
        xt: history buffer
        x: current state
        t: current time index
        p: delay in time steps
        """
        tau_steps = p
        x_delayed = xt[t - tau_steps]
        return -x_delayed
    
    # Setup
    dt = 0.01
    tau = 1.0
    tau_steps = int(tau / dt)
    t_max = 15.0
    n_steps = int(t_max / dt)
    
    # Create history buffer
    buffer_size = tau_steps + 1 + n_steps
    history = jnp.ones(buffer_size)
    history = history.at[tau_steps + 1:].set(0.0)
    
    # Solve
    step, loop = make_dde(dt, tau_steps, dfun)
    buf_final, x = loop(history, tau_steps, t=0)
    
    ts = jnp.arange(0, t_max, dt)
    
    print(f"Delay τ: {tau} time units ({tau_steps} steps)")
    print(f"Initial value: {history[tau_steps]:.4f}")
    print(f"Final value: {x[-1]:.4f}")
    print(f"Solution shows oscillatory decay as expected")
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(ts, x)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title(f'Delay Differential Equation: dx/dt = -x(t-{tau})')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(ts[:-tau_steps], x[:-tau_steps], label='x(t)')
    plt.plot(ts[:-tau_steps], x[tau_steps:], label=f'x(t-τ)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Current State vs Delayed State')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path+'/example4_dde.png', dpi=150)
    print(f"Plot saved as '{path}/example4_dde.png'")
    plt.close()
    
    return x


# =============================================================================
# EXAMPLE 5: Comparing Integration Methods
# =============================================================================
def example_5_method_comparison():
    """Compare different integration methods"""
    print("\n=== Example 5: Comparing Integration Methods ===")
    
    def dfun(x, p):
        """Nonlinear ODE: dx/dt = -x + x^3"""
        return -x + x**3
    
    # Setup
    dt = 0.1  # Larger dt to see differences
    t_max = 5.0
    ts = jnp.arange(0, t_max, dt)
    x0 = 0.5
    
    methods = ['euler', 'heun', 'rk4']
    results = {}
    
    for method in methods:
        step, loop = make_ode(dt, dfun, method=method)
        results[method] = loop(x0, ts, None)
    
    # Reference solution with small dt
    dt_ref = 0.001
    ts_ref = jnp.arange(0, t_max, dt_ref)
    step_ref, loop_ref = make_ode(dt_ref, dfun, method='rk4')
    x_ref = loop_ref(x0, ts_ref, None)
    
    # Calculate errors
    print(f"Time step: dt = {dt}")
    print(f"Reference: dt = {dt_ref} (RK4)\n")
    
    for method in methods:
        # Interpolate reference to compare
        x_ref_interp = jnp.interp(ts, ts_ref, x_ref)
        error = jnp.max(jnp.abs(results[method] - x_ref_interp))
        print(f"{method.upper():6s}: max error = {error:.2e}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for method in methods:
        plt.plot(ts, results[method], 'o-', label=method.upper(), markersize=4)
    plt.plot(ts_ref, x_ref, 'k-', alpha=0.3, linewidth=2, label='Reference')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title(f'Method Comparison (dt={dt})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for method in methods:
        x_ref_interp = jnp.interp(ts, ts_ref, x_ref)
        error = jnp.abs(results[method] - x_ref_interp)
        plt.semilogy(ts, error, 'o-', label=method.upper(), markersize=4)
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.title('Integration Error vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path+'/example5_comparison.png', dpi=150)
    print(f"\nPlot saved as '{path}/example5_comparison.png'")
    plt.close()
    
    return results


# =============================================================================
# MAIN: Run all examples
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("JAX LOOPS INTEGRATION LIBRARY - EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    # example_1_simple_ode()
    # example_2_harmonic_oscillator()
    example_3_stochastic()
    example_4_delay()
    example_5_method_comparison()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 70)
