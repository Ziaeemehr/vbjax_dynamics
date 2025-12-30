"""
Toy Example for Stochastic Delay Differential Equations (SDDEs)
===============================================================

This example demonstrates how to use make_sdde to solve delay differential equations
and stochastic delay differential equations.
"""

import os
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from vbjax_dynamics.loops import make_sdde, randn

jax.config.update("jax_enable_x64", True)
os.makedirs("outputs", exist_ok=True)

# =============================================================================
# EXAMPLE 1: Simple Delay Differential Equation (DDE)
# =============================================================================
def example_delayed_exponential():
    """
    Solve the delayed exponential equation: dx/dt = -x(t-τ)
    
    This is a simple delay differential equation where the derivative at time t
    depends on the state at time t-τ (delayed by τ time units).
    """
    print("\n=== Example 1: Delayed Exponential DDE ===")
    
    # Parameters
    dt = 0.01  # time step
    tau = 1.0  # delay time
    nh = int(tau / dt)  # delay in steps (must be integer)
    t_max = 5.0
    n_steps = int(t_max / dt)
    
    # Initial conditions: constant history for t < 0
    x0 = 1.0
    # Buffer format: [history..., current_state, noise_samples...]
    # We need nh steps of history + current state + noise samples for remaining steps
    buf_size = nh + 1 + n_steps  # history + current + future noise
    buf = jnp.ones(buf_size) * x0  # start with constant history
    
    # Drift function: dx/dt = -x(t-τ)
    # dfun(xt, x, t, p) where:
    # - xt: full history buffer
    # - x: current state (xt[nh+t])
    # - t: time index in buffer
    # - p: parameters (unused here)
    def dfun(xt, x, t, p):
        delayed_x = xt[t - nh]  # x at time t-τ
        return -delayed_x
    
    # Create SDDE integrator (gfun=0 for deterministic DDE)
    step, loop = make_sdde(dt, nh, dfun, gfun=0.0)
    
    # Integrate
    final_buf, trajectory = loop(buf, None)
    
    # Extract the solution (last n_steps of the trajectory)
    solution = trajectory[-n_steps:]
    ts = jnp.arange(0, t_max, dt)
    
    print(f"Initial value: {x0}")
    print(f"Final value: {solution[-1]:.6f}")
    print(f"Buffer size: {buf_size}, nh: {nh}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ts, solution, 'b-', linewidth=2, label='Numerical solution')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Delayed Exponential Equation: dx/dt = -x(t-τ)')
    plt.grid(True)
    plt.legend()
    
    # Plot phase portrait (x(t) vs x(t-τ))
    plt.subplot(2, 1, 2)
    delayed_solution = jnp.roll(solution, nh)  # shift by τ steps
    delayed_solution = delayed_solution[nh:]  # remove invalid initial values
    current_solution = solution[nh:]
    plt.plot(delayed_solution, current_solution, 'r-', alpha=0.7)
    plt.xlabel('x(t-τ)')
    plt.ylabel('x(t)')
    plt.title('Phase Portrait')
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('outputs/example_sdde_dde.png', dpi=150, bbox_inches='tight')
    # plt.show()
    
    return solution


# =============================================================================
# EXAMPLE 2: Stochastic Delay Differential Equation (SDDE)
# =============================================================================
def example_stochastic_delay():
    """
    Solve the stochastic delayed equation: dx/dt = -x(t-τ) + σ*x(t)*dW/dt
    
    This adds multiplicative noise to the delayed exponential equation.
    """
    print("\n=== Example 2: Stochastic Delay SDDE ===")
    
    # Parameters
    dt = 0.01
    tau = 0.5
    nh = int(tau / dt)
    sigma = 0.1  # noise strength
    t_max = 3.0
    n_steps = int(t_max / dt)
    n_trajectories = 5  # run multiple trajectories
    
    # Initial conditions
    x0 = 1.0
    buf_size = nh + 1 + n_steps
    ts = jnp.arange(0, t_max, dt)
    
    # Drift function: dx/dt = -x(t-τ)
    def dfun(xt, x, t, p):
        delayed_x = xt[t - nh]
        return -delayed_x
    
    # Diffusion function: multiplicative noise σ*x
    def gfun(x, p):
        return sigma * x
    
    # Create SDDE integrator
    step, loop = make_sdde(dt, nh, dfun, gfun)
    
    # Run multiple trajectories
    key = random.PRNGKey(42)
    all_trajectories = []
    
    for i in range(n_trajectories):
        # Generate noise samples
        noise_key, key = random.split(key)
        noise_samples = randn(n_steps, key=noise_key)
        
        # Setup buffer with noise
        buf = jnp.ones(buf_size) * x0
        buf = buf.at[nh+1:].set(noise_samples)  # insert noise samples
        
        # Integrate
        final_buf, trajectory = loop(buf, None)
        solution = trajectory[-n_steps:]
        all_trajectories.append(solution)
    
    all_trajectories = jnp.array(all_trajectories)
    
    print(f"Ran {n_trajectories} trajectories")
    print(f"Mean final value: {jnp.mean(all_trajectories[:, -1]):.6f}")
    print(f"Std final value: {jnp.std(all_trajectories[:, -1]):.6f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    for i, traj in enumerate(all_trajectories):
        plt.plot(ts, traj, alpha=0.7, label=f'Trajectory {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title(f'Stochastic Delayed Equation: dx/dt = -x(t-τ) + {sigma}*x(t)*dW/dt')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/example_sdde_stochastic.png', dpi=150, bbox_inches='tight')
    # plt.show()
    
    return all_trajectories


# =============================================================================
# EXAMPLE 3: Logistic Delay Equation
# =============================================================================
def example_logistic_delay():
    """
    Solve the delayed logistic equation: dx/dt = r*x(t)*(1 - x(t-τ)/K)
    
    This models population growth with a delay in the carrying capacity feedback.
    """
    print("\n=== Example 3: Delayed Logistic Equation ===")
    
    # Parameters
    dt = 0.01
    tau = 0.8  # delay
    nh = int(tau / dt)
    r = 1.0    # growth rate
    K = 2.0    # carrying capacity
    t_max = 10.0
    n_steps = int(t_max / dt)
    
    # Initial conditions (start from small population)
    x0 = 0.1
    buf_size = nh + 1 + n_steps
    buf = jnp.ones(buf_size) * x0
    ts = jnp.arange(0, t_max, dt)
    
    # Drift function: dx/dt = r*x*(1 - x(t-τ)/K)
    def dfun(xt, x, t, p):
        delayed_x = xt[t - nh]
        return r * x * (1 - delayed_x / K)
    
    # Create DDE integrator
    step, loop = make_sdde(dt, nh, dfun, gfun=0.0)
    
    # Integrate
    final_buf, trajectory = loop(buf, None)
    solution = trajectory[-n_steps:]
    
    print(f"Initial population: {x0}")
    print(f"Final population: {solution[-1]:.6f}")
    print(f"Carrying capacity: {K}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ts, solution, 'g-', linewidth=2, label='Population')
    plt.axhline(y=K, color='r', linestyle='--', alpha=0.7, label='Carrying capacity')
    plt.xlabel('Time')
    plt.ylabel('Population x(t)')
    plt.title('Delayed Logistic Growth: dx/dt = r*x*(1 - x(t-τ)/K)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/example_sdde_logistic.png', dpi=150, bbox_inches='tight')
    # plt.show()
    
    return solution


if __name__ == "__main__":
    # Run all examples
    sol1 = example_delayed_exponential()
    sol2 = example_stochastic_delay()
    sol3 = example_logistic_delay()
    
    print("\n=== All SDDE Examples Completed ===")