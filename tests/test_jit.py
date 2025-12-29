"""
Test JIT compilation for make_sde_auto
"""

import jax
import jax.numpy as jnp
from jax import random
import time
from vbjax_dynamics.loops import make_sde_auto

jax.config.update("jax_enable_x64", True)


def test_jit_compilation_speedup():
    """Test that JIT compilation provides speedup on repeated calls"""
    
    def drift(x, p):
        return -p * x

    def diffusion(x, p):
        return 0.5

    dt = 0.01
    n_steps = 1000
    x0 = 2.0
    params = 1.0

    step, loop = make_sde_auto(dt, drift, diffusion)

    # First run (includes compilation)
    key = random.PRNGKey(42)
    start = time.time()
    x1 = loop(x0, n_steps, params, key)
    jax.block_until_ready(x1)
    time1 = time.time() - start

    # Second run (compiled)
    key = random.PRNGKey(43)
    start = time.time()
    x2 = loop(x0, n_steps, params, key)
    jax.block_until_ready(x2)
    time2 = time.time() - start

    # Third run (compiled)
    key = random.PRNGKey(44)
    start = time.time()
    x3 = loop(x0, n_steps, params, key)
    jax.block_until_ready(x3)
    time3 = time.time() - start

    avg_compiled_time = (time2 + time3) / 2
    
    # Assert that compilation overhead is significant (first run is slower)
    assert time1 > avg_compiled_time, \
        "First run should be slower due to compilation overhead"
    
    # Assert that results are finite and reasonable
    assert jnp.isfinite(x1[-1]) and jnp.isfinite(x2[-1]) and jnp.isfinite(x3[-1]), \
        "All results should be finite"


def test_jit_with_vmap():
    """Test that JIT works correctly with vmap for parallel trajectories"""
    
    def drift(x, p):
        return -p * x

    def diffusion(x, p):
        return 0.5

    dt = 0.01
    n_steps = 1000
    x0 = 2.0
    params = 1.0
    n_traj = 100

    step, loop = make_sde_auto(dt, drift, diffusion)
    
    keys = random.split(random.PRNGKey(42), n_traj)
    loop_vmap = jax.vmap(lambda k: loop(x0, n_steps, params, k))

    # First run with vmap (includes compilation)
    start = time.time()
    trajectories1 = loop_vmap(keys)
    jax.block_until_ready(trajectories1)
    time_vmap1 = time.time() - start

    # Second run with vmap (compiled)
    start = time.time()
    trajectories2 = loop_vmap(keys)
    jax.block_until_ready(trajectories2)
    time_vmap2 = time.time() - start

    # Assert speedup from compilation
    assert time_vmap1 > time_vmap2, \
        "First vmap run should be slower due to compilation"
    
    # Assert correct shape
    assert trajectories1.shape == (n_traj, n_steps), \
        f"Expected shape ({n_traj}, {n_steps}), got {trajectories1.shape}"
    
    # Assert all results are finite
    assert jnp.all(jnp.isfinite(trajectories1)), \
        "All trajectory values should be finite"


def test_jit_determinism():
    """Test that JIT compilation preserves determinism with same key"""
    
    def drift(x, p):
        return -p * x

    def diffusion(x, p):
        return 0.5

    dt = 0.01
    n_steps = 100
    x0 = 2.0
    params = 1.0

    step, loop = make_sde_auto(dt, drift, diffusion)
    
    # Run twice with the same key
    key = random.PRNGKey(42)
    x1 = loop(x0, n_steps, params, key)
    jax.block_until_ready(x1)
    
    x2 = loop(x0, n_steps, params, key)
    jax.block_until_ready(x2)
    
    # Results should be identical with same key
    assert jnp.allclose(x1, x2), \
        "Same key should produce identical results"
