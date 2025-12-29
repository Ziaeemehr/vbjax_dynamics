"""
Tests for parallel SDE integration using vmap
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
import pytest
from vbjax_dynamics.loops import make_sde, make_sde_auto

jax.config.update("jax_enable_x64", True)


# Define test SDEs
def ornstein_uhlenbeck_drift(x, p):
    """Drift for OU process: dx = -θx dt + σ dW"""
    theta, sigma = p
    return -theta * x


def ornstein_uhlenbeck_diffusion(x, p):
    """Diffusion for OU process"""
    theta, sigma = p
    return sigma


def geometric_brownian_drift(x, p):
    """Drift for GBM: dx = μx dt + σx dW"""
    mu, sigma = p
    return mu * x


def geometric_brownian_diffusion(x, p):
    """Diffusion for GBM"""
    mu, sigma = p
    return sigma * x


class TestVmapMakeSDE:
    """Tests for vmap with make_sde (pre-generated noise)"""
    
    def test_vmap_basic_functionality(self):
        """Test basic vmap functionality with make_sde"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        step, loop = make_sde(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # Generate noise for all trajectories
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        all_noise = vmap(lambda k: random.normal(k, (n_steps,)))(keys)
        
        # Vectorize over noise arrays
        loop_vmap = vmap(lambda zs: loop(x0, zs, params))
        trajectories = loop_vmap(all_noise)
        
        # Check output shape
        assert trajectories.shape == (n_traj, n_steps)
        
        # Check all values are finite
        assert jnp.all(jnp.isfinite(trajectories))
    
    def test_vmap_reproducibility_with_same_noise(self):
        """Test that same noise produces identical results"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        step, loop = make_sde(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # Generate noise once
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        all_noise = vmap(lambda k: random.normal(k, (n_steps,)))(keys)
        
        loop_vmap = vmap(lambda zs: loop(x0, zs, params))
        
        # Run twice with same noise
        trajectories1 = loop_vmap(all_noise)
        trajectories2 = loop_vmap(all_noise)
        
        # Should be identical
        assert jnp.allclose(trajectories1, trajectories2)
        assert jnp.max(jnp.abs(trajectories1 - trajectories2)) == 0.0
    
    def test_vmap_different_initial_conditions(self):
        """Test vmap with different initial conditions for each trajectory"""
        dt = 0.01
        n_steps = 100
        n_traj = 10
        params = (1.0, 0.5)
        
        step, loop = make_sde(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # Different initial conditions
        x0_array = jnp.linspace(0.5, 2.5, n_traj)
        
        # Generate noise
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        all_noise = vmap(lambda k: random.normal(k, (n_steps,)))(keys)
        
        # Vectorize over both x0 and noise
        loop_vmap = vmap(lambda x0, zs: loop(x0, zs, params))
        trajectories = loop_vmap(x0_array, all_noise)
        
        assert trajectories.shape == (n_traj, n_steps)
        
        # Each trajectory should have different behavior based on different x0
        # Check that trajectories are distinct (not all the same)
        assert not jnp.allclose(trajectories[0], trajectories[-1])
    
    def test_vmap_different_parameters(self):
        """Test vmap with different parameters for each trajectory"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        n_traj = 10
        
        step, loop = make_sde(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # Different parameters (theta, sigma) for each trajectory
        theta_array = jnp.linspace(0.5, 2.0, n_traj)
        sigma_array = jnp.full(n_traj, 0.5)
        params_array = jnp.stack([theta_array, sigma_array], axis=1)
        
        # Generate noise
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        all_noise = vmap(lambda k: random.normal(k, (n_steps,)))(keys)
        
        # Vectorize over params and noise
        loop_vmap = vmap(lambda p, zs: loop(x0, zs, p))
        trajectories = loop_vmap(params_array, all_noise)
        
        assert trajectories.shape == (n_traj, n_steps)
        assert jnp.all(jnp.isfinite(trajectories))


class TestVmapMakeSDEAuto:
    """Tests for vmap with make_sde_auto (automatic noise)"""
    
    def test_vmap_basic_functionality(self):
        """Test basic vmap functionality with make_sde_auto"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # Generate keys for each trajectory
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        # Vectorize over random keys
        loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
        trajectories = loop_vmap(keys)
        
        # Check output shape
        assert trajectories.shape == (n_traj, n_steps)
        
        # Check all values are finite
        assert jnp.all(jnp.isfinite(trajectories))
    
    def test_vmap_reproducibility_with_same_seed(self):
        """Test that same seed produces identical results"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
        
        # Run twice with same seed
        key1 = random.PRNGKey(42)
        keys1 = random.split(key1, n_traj)
        trajectories1 = loop_vmap(keys1)
        
        key2 = random.PRNGKey(42)  # Same seed
        keys2 = random.split(key2, n_traj)
        trajectories2 = loop_vmap(keys2)
        
        # Should be identical
        assert jnp.allclose(trajectories1, trajectories2)
        assert jnp.max(jnp.abs(trajectories1 - trajectories2)) == 0.0
    
    def test_vmap_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
        
        # Run with different seeds
        key1 = random.PRNGKey(42)
        keys1 = random.split(key1, n_traj)
        trajectories1 = loop_vmap(keys1)
        
        key2 = random.PRNGKey(123)  # Different seed
        keys2 = random.split(key2, n_traj)
        trajectories2 = loop_vmap(keys2)
        
        # Should be different
        assert not jnp.allclose(trajectories1, trajectories2)
        assert jnp.max(jnp.abs(trajectories1 - trajectories2)) > 0.1
    
    def test_vmap_different_initial_conditions(self):
        """Test vmap with different initial conditions for each trajectory"""
        dt = 0.01
        n_steps = 100
        n_traj = 10
        params = (1.0, 0.5)
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # Different initial conditions
        x0_array = jnp.linspace(0.5, 2.5, n_traj)
        
        # Generate keys
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        # Vectorize over both x0 and keys
        loop_vmap = vmap(lambda x0, k: loop(x0, n_steps, params, k))
        trajectories = loop_vmap(x0_array, keys)
        
        assert trajectories.shape == (n_traj, n_steps)
        
        # Each trajectory should have different behavior based on different x0
        # Check that trajectories are distinct (not all the same)
        assert not jnp.allclose(trajectories[0], trajectories[-1])
    
    def test_vmap_different_parameters(self):
        """Test vmap with different parameters for each trajectory"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        n_traj = 10
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # Different parameters for each trajectory
        theta_array = jnp.linspace(0.5, 2.0, n_traj)
        sigma_array = jnp.full(n_traj, 0.5)
        params_array = jnp.stack([theta_array, sigma_array], axis=1)
        
        # Generate keys
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        # Vectorize over params and keys
        loop_vmap = vmap(lambda p, k: loop(x0, n_steps, p, k))
        trajectories = loop_vmap(params_array, keys)
        
        assert trajectories.shape == (n_traj, n_steps)
        assert jnp.all(jnp.isfinite(trajectories))
    
    def test_vmap_large_ensemble(self):
        """Test vmap with large number of trajectories"""
        dt = 0.01
        n_steps = 1000  # Need more steps for convergence to stationary distribution
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 1000
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
        trajectories = loop_vmap(keys)
        
        assert trajectories.shape == (n_traj, n_steps)
        assert jnp.all(jnp.isfinite(trajectories))
        
        # Check ensemble statistics (OU process should converge to mean=0)
        final_values = trajectories[:, -1]
        ensemble_mean = jnp.mean(final_values)
        
        # Mean should be close to 0 (with reasonable tolerance for stochastic process)
        assert jnp.abs(ensemble_mean) < 0.2


class TestVmapComparison:
    """Compare make_sde and make_sde_auto with vmap"""
    
    def test_both_approaches_produce_same_results(self):
        """Test that both approaches give identical results with same random state"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        # Approach 1: make_sde with pre-generated noise
        step1, loop1 = make_sde(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        all_noise = vmap(lambda k: random.normal(k, (n_steps,)))(keys)
        loop1_vmap = vmap(lambda zs: loop1(x0, zs, params))
        trajectories1 = loop1_vmap(all_noise)
        
        # Approach 2: make_sde_auto
        step2, loop2 = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        key = random.PRNGKey(42)  # Same seed
        keys = random.split(key, n_traj)
        loop2_vmap = vmap(lambda k: loop2(x0, n_steps, params, k))
        trajectories2 = loop2_vmap(keys)
        
        # Should produce identical results
        assert jnp.allclose(trajectories1, trajectories2, rtol=1e-10)
    
    def test_ensemble_statistics_match(self):
        """Test that ensemble statistics match between approaches"""
        dt = 0.01
        n_steps = 500
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 500
        
        # Approach 1: make_sde
        step1, loop1 = make_sde(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        all_noise = vmap(lambda k: random.normal(k, (n_steps,)))(keys)
        loop1_vmap = vmap(lambda zs: loop1(x0, zs, params))
        trajectories1 = loop1_vmap(all_noise)
        
        # Approach 2: make_sde_auto
        step2, loop2 = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        key = random.PRNGKey(123)  # Different seed for statistical comparison
        keys = random.split(key, n_traj)
        loop2_vmap = vmap(lambda k: loop2(x0, n_steps, params, k))
        trajectories2 = loop2_vmap(keys)
        
        # Compute ensemble statistics
        mean1 = jnp.mean(trajectories1[:, -1])
        std1 = jnp.std(trajectories1[:, -1])
        
        mean2 = jnp.mean(trajectories2[:, -1])
        std2 = jnp.std(trajectories2[:, -1])
        
        # Statistics should be similar (within sampling error)
        assert jnp.abs(mean1 - mean2) < 0.2
        assert jnp.abs(std1 - std2) < 0.1


class TestVmapComplexScenarios:
    """Test vmap with more complex scenarios"""
    
    def test_vmap_geometric_brownian_motion(self):
        """Test vmap with geometric Brownian motion (multiplicative noise)"""
        dt = 0.01
        n_steps = 100
        x0 = 1.0
        params = (0.05, 0.2)  # mu=0.05, sigma=0.2
        n_traj = 50
        
        step, loop = make_sde_auto(dt, geometric_brownian_drift, geometric_brownian_diffusion)
        
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
        trajectories = loop_vmap(keys)
        
        # GBM stays positive
        assert jnp.all(trajectories > 0)
        
        # Check shape
        assert trajectories.shape == (n_traj, n_steps)
    
    def test_vmap_nested_vectorization(self):
        """Test nested vmap for grid of parameters and initial conditions"""
        dt = 0.01
        n_steps = 50
        n_x0 = 5
        n_theta = 4
        n_traj_per_combo = 10
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # Grid of initial conditions and parameters
        x0_array = jnp.linspace(0.5, 2.5, n_x0)
        theta_array = jnp.linspace(0.5, 2.0, n_theta)
        sigma = 0.5
        
        key = random.PRNGKey(42)
        
        results = []
        for x0 in x0_array:
            for theta in theta_array:
                params = (theta, sigma)
                keys = random.split(key, n_traj_per_combo)
                key = keys[0]  # Update key for next iteration
                
                loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
                trajectories = loop_vmap(keys[1:])
                results.append(trajectories)
        
        # Should have results for all combinations
        assert len(results) == n_x0 * n_theta
        
        # Each should have correct shape
        for traj in results:
            assert traj.shape == (n_traj_per_combo - 1, n_steps)
    
    def test_vmap_with_jit_compilation(self):
        """Test that vmap works correctly with JIT compilation"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        # JIT compile the vmapped function
        @jax.jit
        def run_ensemble(keys):
            loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
            return loop_vmap(keys)
        
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        # First call (compilation)
        trajectories1 = run_ensemble(keys)
        
        # Second call (should use compiled version)
        trajectories2 = run_ensemble(keys)
        
        # Should be identical
        assert jnp.allclose(trajectories1, trajectories2)
        assert trajectories1.shape == (n_traj, n_steps)


class TestVmapEdgeCases:
    """Test edge cases and potential issues"""
    
    def test_vmap_single_trajectory(self):
        """Test vmap with single trajectory (edge case)"""
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 1
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
        trajectories = loop_vmap(keys)
        
        assert trajectories.shape == (1, n_steps)
        assert jnp.all(jnp.isfinite(trajectories))
    
    def test_vmap_very_small_dt(self):
        """Test vmap with very small time step"""
        dt = 0.0001
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
        
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
        trajectories = loop_vmap(keys)
        
        assert trajectories.shape == (n_traj, n_steps)
        assert jnp.all(jnp.isfinite(trajectories))
    
    def test_vmap_zero_diffusion(self):
        """Test vmap with zero diffusion (deterministic ODE)"""
        def zero_diffusion(x, p):
            return 0.0
        
        dt = 0.01
        n_steps = 100
        x0 = 2.0
        params = (1.0, 0.5)
        n_traj = 10
        
        step, loop = make_sde_auto(dt, ornstein_uhlenbeck_drift, zero_diffusion)
        
        key = random.PRNGKey(42)
        keys = random.split(key, n_traj)
        
        loop_vmap = vmap(lambda k: loop(x0, n_steps, params, k))
        trajectories = loop_vmap(keys)
        
        # All trajectories should be identical (deterministic)
        for i in range(1, n_traj):
            assert jnp.allclose(trajectories[0], trajectories[i])
