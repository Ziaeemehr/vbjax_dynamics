#!/usr/bin/env python
"""
Minimal example demonstrating vbjax_dynamics configuration utilities

This script shows how to use the precision configuration functions.
"""

import sys
from pathlib import Path

# Add src directory to path so we can import without installing
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

import jax
import jax.numpy as jnp
from vbjax_dynamics import configure_jax, precision_context, print_jax_config

print("="*70)
print("MINIMAL CONFIGURATION EXAMPLE")
print("="*70)

# =============================================================================
# Example 1: Check default configuration
# =============================================================================
print("\n1. Default Configuration (32-bit)")
print("-"*70)
print_jax_config()

x = jnp.array([1.0, 2.0, 3.0])
print(f"Array dtype: {x.dtype}")
print(f"Array values: {x}")

# =============================================================================
# Example 2: Use configure_jax for global 64-bit precision
# =============================================================================
print("\n2. Enable 64-bit Precision Globally")
print("-"*70)
configure_jax(enable_x64=True)

x64 = jnp.array([1.0, 2.0, 3.0])
print(f"Array dtype: {x64.dtype}")
print(f"Array values: {x64}")

# Simple computation to show precision difference
result = jnp.sum(x64 ** 2)
print(f"Sum of squares: {result} (dtype: {result.dtype})")

# =============================================================================
# Example 3: Use precision_context for temporary precision change
# =============================================================================
print("\n3. Temporary Precision Change with Context Manager")
print("-"*70)

# First, disable x64 to demonstrate the context manager
jax.config.update("jax_enable_x64", False)
print("Current precision: 32-bit")

x32 = jnp.array([1.0, 2.0, 3.0])
print(f"Array dtype outside context: {x32.dtype}")

# Use context manager for high-precision computation
with precision_context(enable_x64=True):
    print("\nInside precision_context(enable_x64=True):")
    x64_temp = jnp.array([1.0, 2.0, 3.0])
    print(f"  Array dtype: {x64_temp.dtype}")
    
    # High precision computation
    result_hp = jnp.sqrt(jnp.sum(x64_temp ** 2))
    print(f"  ||x||_2 = {result_hp:.15f} (dtype: {result_hp.dtype})")

print("\nAfter exiting context:")
x32_again = jnp.array([1.0, 2.0, 3.0])
print(f"Array dtype: {x32_again.dtype}")

# =============================================================================
# Example 4: Precision matters for numerical accuracy
# =============================================================================
print("\n4. Precision Comparison")
print("-"*70)

# A computation that shows precision differences
# Computing: (1 + 1e-10) - 1
epsilon = 1e-10

# 32-bit
jax.config.update("jax_enable_x64", False)
val32 = jnp.float32(1.0) + jnp.float32(epsilon)
diff32 = val32 - jnp.float32(1.0)
print(f"32-bit: (1 + {epsilon}) - 1 = {diff32} (expected: {epsilon})")

# 64-bit
jax.config.update("jax_enable_x64", True)
val64 = jnp.float64(1.0) + jnp.float64(epsilon)
diff64 = val64 - jnp.float64(1.0)
print(f"64-bit: (1 + {epsilon}) - 1 = {diff64} (expected: {epsilon})")

print(f"\nRelative error (32-bit): {abs(diff32 - epsilon) / epsilon * 100:.1f}%")
print(f"Relative error (64-bit): {abs(diff64 - epsilon) / epsilon * 100:.1f}%")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Three ways to configure precision:

1. configure_jax(enable_x64=True)
   - Sets precision for entire session
   - Simple one-time setup
   - Good for scripts that need consistent precision

2. precision_context(enable_x64=True)
   - Temporary precision change
   - Automatically restores original setting
   - Good for specific high-precision computations

3. Manual JAX configuration (before importing)
   import jax
   jax.config.update("jax_enable_x64", True)
   import vbjax_dynamics
   - Full control
   - Standard JAX approach
   
Recommendations:
- Use 32-bit (default) for most ML/prototyping work
- Use 64-bit for high-accuracy scientific computing
- Use context manager when mixing precision requirements
""")

print("\n✓ Example completed successfully!")
