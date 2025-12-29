# SDE Noise Generation Options

## Two Ways to Use `make_sde`

When solving Stochastic Differential Equations (SDEs), you have two options for handling noise generation:

---

## Option 1: Pre-generate Noise (Original `make_sde`)

### Usage
```python
from jax import random
from vbjax_dydomics.loops import make_sde

# Setup
step, loop = make_sde(dt, drift_func, diffusion_func)

# Generate noise BEFORE integration
key = random.PRNGKey(42)
n_steps = int(t_max / dt)
zs = random.normal(key, (n_steps,))

# Integrate with pre-generated noise
x = loop(x0, zs, params)
```

### When to Use
✅ **Use this when you need:**
- Custom noise distributions (non-Gaussian)
- To reuse the same noise for multiple runs
- To save/load noise for reproducibility
- Fine control over noise generation
- Advanced research applications

### Advantages
- Full control over noise
- Can use any noise distribution
- Can save noise for reproducibility
- More flexible for research

### Disadvantages
- Extra step to generate noise
- More verbose code
- Need to manage noise array

---

## Option 2: Automatic Noise Generation (New `make_sde_auto`)

### Usage
```python
from jax import random
from vbjax_dydomics.loops import make_sde_auto

# Setup with automatic noise
step, loop = make_sde_auto(dt, drift_func, diffusion_func)

# Integrate - noise generated internally!
key = random.PRNGKey(42)
n_steps = int(t_max / dt)
x = loop(x0, n_steps, params, key)
```

### When to Use
✅ **Use this when you want:**
- Simpler, cleaner code
- Standard Gaussian noise
- Quick prototyping
- Teaching or learning
- Less memory management

### Advantages
- Simpler API (one less step)
- Cleaner, more readable code
- Good for beginners
- Less to manage

### Disadvantages
- Only Gaussian noise
- Less control over noise
- Can't easily save/reuse noise

---

## Comparison Example

### Full Example: Ornstein-Uhlenbeck Process

```python
import jax.numpy as jnp
from jax import random
from vbjax_dydomics.loops import make_sde, make_sde_auto

# Define the SDE: dx = -θx dt + σ dW
def drift(x, p):
    theta, sigma = p
    return -theta * x

def diffusion(x, p):
    theta, sigma = p
    return sigma

# Parameters
dt = 0.01
t_max = 10.0
n_steps = int(t_max / dt)
x0 = 2.0
params = (1.0, 0.5)  # (theta, sigma)

# ========================================
# OPTION 1: Pre-generate noise
# ========================================
print("Option 1: Pre-generate noise")
step1, loop1 = make_sde(dt, drift, diffusion)

key = random.PRNGKey(42)
zs = random.normal(key, (n_steps,))
x1 = loop1(x0, zs, params)

print(f"Final value: {x1[-1]:.6f}")

# ========================================
# OPTION 2: Automatic noise
# ========================================
print("\nOption 2: Automatic noise")
step2, loop2 = make_sde_auto(dt, drift, diffusion)

key = random.PRNGKey(42)  # Same key!
x2 = loop2(x0, n_steps, params, key)

print(f"Final value: {x2[-1]:.6f}")

# Verify they're identical
print(f"\nAre they identical? {jnp.allclose(x1, x2)}")
print(f"Max difference: {jnp.max(jnp.abs(x1 - x2)):.2e}")
```

---

## Multiple Trajectories

### Option 1: Pre-generate
```python
n_traj = 100
keys = random.split(key, n_traj)

trajectories = []
for k in keys:
    zs = random.normal(k, (n_steps,))
    x = loop1(x0, zs, params)
    trajectories.append(x)
```

### Option 2: Automatic
```python
n_traj = 100
keys = random.split(key, n_traj)

trajectories = []
for k in keys:
    x = loop2(x0, n_steps, params, k)
    trajectories.append(x)
```

**Even simpler with `vmap`:**
```python
import jax

# Vectorize over random keys
loop_vmap = jax.vmap(lambda k: loop2(x0, n_steps, params, k))
trajectories = loop_vmap(keys)
# Shape: (n_traj, n_steps)
```

---

## Advanced: Custom Noise (Only Option 1)

If you need non-Gaussian noise, you must use Option 1:

```python
# Example: Uniform noise instead of Gaussian
zs = random.uniform(key, (n_steps,), minval=-1, maxval=1)
x = loop1(x0, zs, params)

# Example: Saved noise for reproducibility
import numpy as np
zs = jnp.load('my_noise.npy')
x = loop1(x0, zs, params)

# Example: Correlated noise
# Generate colored noise with specific spectrum
zs = generate_colored_noise(key, n_steps, alpha=1.0)
x = loop1(x0, zs, params)
```

---

## Performance Comparison

Both approaches have **identical performance**:
- Both are JIT-compiled
- Noise generation is fast in both cases
- Memory usage is similar
- No significant speed difference

---

## Reproducibility

### Option 1: Easy to save and reload
```python
# Save noise for reproducibility
zs = random.normal(key, (n_steps,))
jnp.save('noise.npy', zs)

# Later...
zs = jnp.load('noise.npy')
x = loop1(x0, zs, params)
```

### Option 2: Use same key
```python
# Always use the same key for reproducibility
key = random.PRNGKey(42)
x = loop2(x0, n_steps, params, key)
```

---

## Recommendations

### For Beginners
→ **Use `make_sde_auto`**
- Simpler API
- Less to remember
- Cleaner code

### For Standard Applications
→ **Use `make_sde_auto`**
- Gaussian noise is usually sufficient
- Code is more readable
- Less boilerplate

### For Research / Advanced Use
→ **Use `make_sde`**
- Custom noise distributions
- Save/load noise for reproducibility
- More control over experiments
- Non-Gaussian processes

### For Teaching
→ **Use `make_sde_auto`**
- Focus on the SDE, not noise management
- Less cognitive load for students
- Cleaner examples

---

## Migration Guide

### From Option 1 to Option 2
```python
# Before (Option 1)
step, loop = make_sde(dt, drift, diffusion)
zs = random.normal(key, (n_steps,))
x = loop(x0, zs, params)

# After (Option 2)
step, loop = make_sde_auto(dt, drift, diffusion)
x = loop(x0, n_steps, params, key)
```

### From Option 2 to Option 1
```python
# Before (Option 2)
step, loop = make_sde_auto(dt, drift, diffusion)
x = loop(x0, n_steps, params, key)

# After (Option 1)
step, loop = make_sde(dt, drift, diffusion)
zs = random.normal(key, (n_steps,))
x = loop(x0, zs, params)
```

---

## Summary

| Feature | Option 1 (Pre-gen) | Option 2 (Auto) |
|---------|-------------------|-----------------|
| Code simplicity | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Flexibility | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Custom noise | ✅ Yes | ❌ No |
| Gaussian noise | ✅ Yes | ✅ Yes |
| Save/reload noise | ✅ Easy | ⚠️ Harder |
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Reproducibility | ✅ Easy | ✅ Easy |
| Best for | Research | Learning/Prototyping |

**Bottom line:** Both are great! Choose based on your needs:
- **Simplicity** → `make_sde_auto`
- **Control** → `make_sde`
