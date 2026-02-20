# Baum-Welch for JAX
Discrete HMM inference with JAX (`baum_welch_jax`)

This package implements the Baum–Welch (EM) algorithm for hidden Markov model (HMM) inference using JAX.
It supports discrete-state, discrete-emission HMMs with optional GPU acceleration.
The code is intentionally minimal and written for readability, with a small number of core functions.

Features and limitations:

- Discrete-state, discrete-emission HMMs only.
- Forward–Backward algorithm for both regular and log probabilities.
- Baum–Welch (EM) implementation both in probability and log-probability space.
- Option to freeze rows of the transition or emission matrices during inference.
- Extensive test coverage.

## Installation

Quick install with `pip install git+https://github.com/tobias-heinzle/BaumWelchJax.git`.
Alternatively, clone the repo and run `pip install .`, and if you want to install the additional dependencies (e.g. cuda) use `pip install .[cuda]`.

- Python >= 3.12
- Dependencies: Only `jax`!

Optional dependency groups are 
- `cuda: jax[cuda]` for GPU support
- `dev: pytest` for running the tests, and
- `notebook: ipykernel, matplotlib` for the interactive notebooks

## Quick start

Here is a quick script demonstrating the basics:

```python
import jax
import jax.numpy as jnp

from baum_welch_jax import (
    HiddenMarkovParameters, 
    baum_welch,
    generate_sequence,
)

# Recommended for best performance!
jax.config.update('jax_enable_x64', True)

# Defining a hidden Markov model
T = jnp.array([[0.6, 0.4], [0.1, 0.9]])
O = jnp.array([[0.7, 0.3], [0.0, 1.0]])
mu = jnp.array([1.0, 0.0])

hmm = HiddenMarkovParameters(T, O, mu)

# Generating a sequence
states, observations = generate_sequence(jax.random.key(0), hmm, length=100)

# Estimating model parameters
initial_guess = HiddenMarkovParameters(
    jnp.array([[0.51, 0.49], [0.49, 0.51]]), 
    jnp.array([[0.51, 0.49], [0.49, 0.51]]), 
    jnp.ones_like(mu) / 2)
    
# Run Baum–Welch until convergence
estimation_result = baum_welch(observations, initial_guess)

print('Iterations:', estimation_result.iterations)
print(estimation_result.params.to_prob())
```

For further examples check out the examples in the `notebooks` directory!

## Repository structure

```
.
├── notebooks/
├── src/
│   └── baum_welch_jax/
│       ├── algorithms/
│       ├── models/
│       └── util/
└── tests/
```

## Documentation

- Function-level documentation is in docstrings
- See `src/baum_welch_jax/algorithms` for implementations
- See `notebooks/` for runnable examples
