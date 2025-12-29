# Tests for vbjax_dynamics

This directory contains tests for the vbjax_dynamics package.

## Test Files

### Unit Tests (for pytest)
- `test_ode.py` - Unit tests for ODE solvers using pytest
- `test_sde.py` - Unit tests for SDE solvers using pytest

### Validation Tests (standalone scripts)
- `test_accuracy.py` - Accuracy validation against analytical solutions and SciPy
- `test_jit.py` - JIT compilation verification
- `test_jit_performance.py` - Performance benchmarks for JIT compilation

## Running Tests

### Run pytest unit tests:
```bash
# From the repository root
pytest tests/

# Run specific test file
pytest tests/test_ode.py

# Run with coverage
pytest --cov=vbjax_dynamics tests/
```

### Run standalone validation scripts:
```bash
# From the repository root
python tests/test_accuracy.py
python tests/test_jit.py
python tests/test_jit_performance.py
```

### Or from the tests directory:
```bash
cd tests
python test_accuracy.py
python test_jit.py
python test_jit_performance.py
```

## Dependencies

Tests require:
- `pytest` (for unit tests)
- `scipy` (for accuracy validation)
- `matplotlib` (optional, for examples)

Install with:
```bash
pip install -e ".[dev]"
```
