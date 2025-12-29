"""
SDE Noise Generation Approaches - Comparison
============================================

This example demonstrates two approaches for handling noise in SDEs:
1. Pre-generating noise (original make_sde)
2. Automatic noise generation (new make_sde_auto)
"""

import os
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from vbjax_dynamics.loops import make_sde, make_sde_auto

jax.config.update("jax_enable_x64", True)
path = "outputs"
os.makedirs(path, exist_ok=True)

print("=" * 70)
print("SDE Noise Generation: Two Approaches")
print("=" * 70)

# Define an Ornstein-Uhlenbeck process
def drift(x, p):
    """Drift: -θx"""
    theta, sigma = p
    return -theta * x

def diffusion(x, p):
    """Diffusion: σ"""
    theta, sigma = p
    return sigma

# Parameters
dt = 0.01
t_max = 10.0
n_steps = int(t_max / dt)
theta = 1.0
sigma = 0.5
params = (theta, sigma)
x0 = 2.0

print(f"\nProblem: Ornstein-Uhlenbeck process")
print(f"  dx = -θx dt + σ dW")
print(f"  θ = {theta}, σ = {sigma}")
print(f"  Initial condition: x(0) = {x0}")
print(f"  Time steps: {n_steps}")

# =============================================================================
# APPROACH 1: Pre-generate noise (original)
# =============================================================================
print("\n" + "=" * 70)
print("APPROACH 1: Pre-generate Noise")
print("=" * 70)

print("""
# Create integrator
step, loop = make_sde(dt, drift, diffusion)

# Generate noise beforehand
key = random.PRNGKey(42)
zs = random.normal(key, (n_steps,))

# Integrate
x = loop(x0, zs, params)
""")

key = random.PRNGKey(42)
zs = random.normal(key, (n_steps,))

step1, loop1 = make_sde(dt, drift, diffusion)
x1 = loop1(x0, zs, params)

print(f"Result: x(T) = {x1[-1]:.6f}")
print(f"Mean (last half): {jnp.mean(x1[n_steps//2:]):.6f}")

# =============================================================================
# APPROACH 2: Automatic noise generation (new)
# =============================================================================
print("\n" + "=" * 70)
print("APPROACH 2: Automatic Noise Generation")
print("=" * 70)

print("""
# Create integrator with auto noise
step, loop = make_sde_auto(dt, drift, diffusion)

# Integrate - noise generated internally!
key = random.PRNGKey(42)
x = loop(x0, n_steps, params, key)
""")

key2 = random.PRNGKey(42)
step2, loop2 = make_sde_auto(dt, drift, diffusion)
x2 = loop2(x0, n_steps, params, key2)

print(f"Result: x(T) = {x2[-1]:.6f}")
print(f"Mean (last half): {jnp.mean(x2[n_steps//2:]):.6f}")

# =============================================================================
# VERIFICATION: Both approaches give identical results
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION: Are they identical?")
print("=" * 70)

max_diff = jnp.max(jnp.abs(x1 - x2))
print(f"\nMaximum difference: {max_diff:.2e}")

if max_diff < 1e-15:
    print("✓ IDENTICAL - Both approaches give the same results!")
else:
    print("✗ DIFFERENT - Results do not match")

# =============================================================================
# COMPARISON: Pros and Cons
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: Pros and Cons")
print("=" * 70)

print("""
APPROACH 1: Pre-generate Noise (make_sde)
------------------------------------------
✓ Pros:
  - More control over noise generation
  - Can use custom noise (not just Gaussian)
  - Can reuse same noise for multiple runs
  - Can save/load noise for reproducibility
  - More flexible for advanced use cases

✗ Cons:
  - Extra step to generate noise
  - Need to manage noise array memory
  - More verbose code

Best for:
  - Custom noise distributions
  - Reproducible simulations (save noise)
  - Advanced research applications


APPROACH 2: Automatic Noise (make_sde_auto)
-------------------------------------------
✓ Pros:
  - Simpler, more concise code
  - Less memory management
  - Cleaner API (just pass key and n_steps)
  - Good for quick prototyping

✗ Cons:
  - Only Gaussian noise
  - Less control over noise generation
  - Can't easily reuse same noise

Best for:
  - Quick prototyping
  - Standard Gaussian noise
  - Simpler code
  - Teaching/learning
""")

# =============================================================================
# ADVANCED: Multiple trajectories comparison
# =============================================================================
print("\n" + "=" * 70)
print("ADVANCED: Multiple Trajectories")
print("=" * 70)

n_traj = 100
print(f"\nGenerating {n_traj} trajectories with each approach...")

# Approach 1: Pre-generate all noise
key = random.PRNGKey(123)
keys = random.split(key, n_traj)

trajectories1 = []
for k in keys:
    zs = random.normal(k, (n_steps,))
    x = loop1(x0, zs, params)
    trajectories1.append(x[-1])

trajectories1 = jnp.array(trajectories1)
mean1 = jnp.mean(trajectories1)
std1 = jnp.std(trajectories1)

# Approach 2: Auto noise generation
trajectories2 = []
for k in keys:
    x = loop2(x0, n_steps, params, k)
    trajectories2.append(x[-1])

trajectories2 = jnp.array(trajectories2)
mean2 = jnp.mean(trajectories2)
std2 = jnp.std(trajectories2)

print(f"\nApproach 1 - Mean: {mean1:.6f}, Std: {std1:.6f}")
print(f"Approach 2 - Mean: {mean2:.6f}, Std: {std2:.6f}")
print(f"Difference - Mean: {abs(mean1-mean2):.2e}, Std: {abs(std1-std2):.2e}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("Creating visualization...")
print("=" * 70)

ts = jnp.arange(0, t_max, dt)

# Generate a few sample trajectories for plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Single trajectory comparison
key_plot = random.PRNGKey(999)
zs_plot = random.normal(key_plot, (n_steps,))
x_method1 = loop1(x0, zs_plot, params)
x_method2 = loop2(x0, n_steps, params, key_plot)

axes[0, 0].plot(ts, x_method1, label='Pre-generated noise', linewidth=2)
axes[0, 0].plot(ts, x_method2, '--', label='Auto noise', linewidth=2, alpha=0.7)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('x(t)')
axes[0, 0].set_title('Single Trajectory: Both Methods')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Multiple trajectories - Approach 1
key_multi = random.PRNGKey(555)
keys_multi = random.split(key_multi, 20)
for k in keys_multi[:20]:
    zs = random.normal(k, (n_steps,))
    x = loop1(x0, zs, params)
    axes[0, 1].plot(ts, x, alpha=0.3, color='blue')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('x(t)')
axes[0, 1].set_title('Approach 1: Pre-generated Noise (20 trajectories)')
axes[0, 1].grid(True)

# Plot 3: Multiple trajectories - Approach 2
for k in keys_multi[:20]:
    x = loop2(x0, n_steps, params, k)
    axes[1, 0].plot(ts, x, alpha=0.3, color='red')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('x(t)')
axes[1, 0].set_title('Approach 2: Auto Noise (20 trajectories)')
axes[1, 0].grid(True)

# Plot 4: Distribution comparison
axes[1, 1].hist(trajectories1, bins=30, alpha=0.5, label='Approach 1', 
                density=True, color='blue')
axes[1, 1].hist(trajectories2, bins=30, alpha=0.5, label='Approach 2', 
                density=True, color='red')
axes[1, 1].set_xlabel(f'x(T={t_max})')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title(f'Final Value Distribution ({n_traj} trajectories)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(path+'/sde_noise_comparison.png', dpi=150)
print("Plot saved as 'sde_noise_comparison.png'")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

print("""
📌 KEY POINTS:

1. Both approaches give IDENTICAL results
   (when using the same random key)

2. Use make_sde (pre-generated noise) when:
   • You need custom noise distributions
   • You want to save/load noise for reproducibility
   • You need fine control over noise generation
   • You're doing research requiring specific noise

3. Use make_sde_auto (automatic noise) when:
   • You want simpler, cleaner code
   • Standard Gaussian noise is sufficient
   • You're prototyping or learning
   • Code readability is priority

4. Performance is essentially identical
   • Both are JIT-compiled
   • Noise generation is fast
   • Memory usage similar for single runs

5. For multiple trajectories:
   • Pre-generate is slightly more explicit
   • Auto is cleaner with proper key splitting
   • Both work well with jax.vmap

✨ RECOMMENDATION:
   Start with make_sde_auto for simplicity.
   Switch to make_sde if you need more control.
""")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)
