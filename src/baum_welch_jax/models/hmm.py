from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HiddenMarkovModel:
    T: Array    # Transition probabilities
    O: Array    # Observation probabilities
    mu: Array   # Initial state probabilities

def check_valid_hmm(hmm: HiddenMarkovModel) -> bool:
    '''JIT save validation for HiddenMarkovModels.
    Returns a bool that indicates if validation was succesful. '''

    correct_dims = jnp.all(jnp.array([
        hmm.T.ndim == 2, 
        hmm.O.ndim == 2, 
        hmm.mu.ndim == 1, 
        hmm.T.shape[0] == hmm.T.shape[1]
    ]))
    
    all_positive = jnp.all(jnp.array([
        jnp.all(hmm.T >= 0),
        jnp.all(hmm.O >= 0),
        jnp.all(hmm.mu >= 0)
    ]))

    all_sum_to_one = jnp.all(jnp.array([
        jnp.allclose(jnp.sum(hmm.T, axis=1), 1.0),
        jnp.allclose(jnp.sum(hmm.O, axis=1), 1.0),
        jnp.allclose(jnp.sum(hmm.mu), 1.0)
    ]))
    
    return jnp.all(jnp.array([correct_dims, all_positive, all_sum_to_one]))

def assert_valid_hmm(hmm: HiddenMarkovModel):
    '''
    Runs assertions for critical properties of a HiddenMarkovModel.
    Throws a ValueError if anything is incorrect.
    '''
    
    if hmm.T.ndim != 2:
        raise ValueError("T must be a 2D probability vector")

    if not jnp.allclose(jnp.sum(hmm.T, axis=1), 1.0):
        raise ValueError("Rows of T must sum to 1")

    if jnp.any(hmm.T < 0):
        raise ValueError("T must be non-negative")
    
    # Shape / normalization checks for O
    if hmm.O.ndim != 2:
        raise ValueError("O must be a 2D probability vector")

    if not jnp.allclose(jnp.sum(hmm.O, axis=1), 1.0):
        raise ValueError("Rows of T must sum to 1")

    if jnp.any(hmm.O < 0):
        raise ValueError("T must be non-negative")

    # Shape / normalization checks for mu
    if hmm.mu.ndim != 1:
        raise ValueError("mu must be a 1D probability vector")

    if not jnp.allclose(jnp.sum(hmm.mu), 1.0):
        raise ValueError("mu must sum to 1")

    if jnp.any(hmm.mu < 0):
        raise ValueError("mu must be non-negative")
