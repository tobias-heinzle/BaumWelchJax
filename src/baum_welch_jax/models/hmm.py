from dataclasses import dataclass, field
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HiddenMarkovParameters:
    '''This class contains the parameters of a hidden Markov model. It is a registered `PyTree` node
    with four data fields:

    `T`     ... transition matrix,  `shape = (n, n)`

    `O`     ... observation matrix, `shape = (n, m)`

    `mu`    ... initial state distributions, `shape = (n,) or (k, n)` where `k` is the number of sequences

    `is_log`... flag indicating if parametes are represented as log probabilities

    Note that initial state probabilities are organized as an array that can contain 
    several distributions, corresponding to different sequences.

    Some convenience methods for converting between log probabilities and
    regular probabilities are also provided.
    '''

    T: Array    # Transition (log)probabilities
    O: Array    # Observation (log)probabilities
    mu: Array   # Initial state (log)probabilities

    # Indicates if probabilities are regular or log probs
    is_log: bool = field(metadata={"static": True}, default=False)

    def to_log(self) -> Self:
        if self.is_log:
            raise ValueError('Only regular probabilities can be transformed to log!')

        return HiddenMarkovParameters(
            jnp.log(self.T),
            jnp.log(self.O),
            jnp.log(self.mu),
            is_log=True
        )
    
    def to_prob(self) -> Self:
        if not self.is_log:
            raise ValueError('Only log probabilities can be transformed to regular!')
            
        return HiddenMarkovParameters(
            jnp.exp(self.T),
            jnp.exp(self.O),
            jnp.exp(self.mu),
            is_log=False
        )
    
    def astype(self, dtype: jnp.floating) -> Self:
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("dtype must be floating point number")
        
        return HiddenMarkovParameters(
            self.T.astype(dtype),
            self.O.astype(dtype),
            self.mu.astype(dtype),
            self.is_log
        )
    
    def replace_mu(self, new_mu: Array) -> Self:
        return HiddenMarkovParameters(
            self.T,
            self.O,
            new_mu,
            self.is_log
        )


def check_valid_hmm(hmm: HiddenMarkovParameters) -> bool:
    '''JIT save validation for HiddenMarkovModels.
    Returns a bool that indicates if validation was succesful. '''

    correct_dims = jnp.all(jnp.array([
        hmm.T.ndim == 2, 
        hmm.O.ndim == 2, 
        hmm.mu.ndim == 1 or hmm.mu.ndim == 2, 
        hmm.T.shape[0] == hmm.T.shape[1]
    ]))

    is_float = (
        jnp.issubdtype(hmm.T.dtype, jnp.floating) 
        & jnp.issubdtype(hmm.O.dtype, jnp.floating) 
        & jnp.issubdtype(hmm.mu.dtype, jnp.floating)
    )
    
    if hmm.is_log:
        all_sum_to_one = jnp.all(jnp.array([
            jnp.allclose(logsumexp(hmm.T, axis=1), 0.0),
            jnp.allclose(logsumexp(hmm.O, axis=1), 0.0),
            jnp.allclose(logsumexp(hmm.mu, axis=-1), 0.0)
        ]))

        return jnp.all(jnp.array([correct_dims, all_sum_to_one]))

    else:
        all_positive = jnp.all(jnp.array([
            jnp.all(hmm.T >= 0),
            jnp.all(hmm.O >= 0),
            jnp.all(hmm.mu >= 0)
        ]))

        all_sum_to_one = jnp.all(jnp.array([
            jnp.allclose(jnp.sum(hmm.T, axis=1), 1.0),
            jnp.allclose(jnp.sum(hmm.O, axis=1), 1.0),
            jnp.allclose(jnp.sum(hmm.mu, axis=-1), 1.0)
        ]))
    
        return jnp.all(jnp.array([correct_dims, all_positive, all_sum_to_one, is_float]))

def assert_valid_hmm(hmm: HiddenMarkovParameters):
    '''
    Runs assertions for critical properties of a HiddenMarkovModel.
    Throws a `ValueError` if anything is incorrect.
    '''

    # Shape checks for O, T, mu
    if hmm.T.ndim != 2:
        raise ValueError("T must be a 2D matrix")
    
    if hmm.O.ndim != 2:
        raise ValueError("O must be a 2D matrix")

    if hmm.mu.ndim > 2 or hmm.mu.ndim < 1:
        raise ValueError("mu must be a either a 1D or 2D array")
    
    if not jnp.issubdtype(hmm.T.dtype, jnp.floating):
        raise ValueError("T.dtype must be floating point number")
    
    if not jnp.issubdtype(hmm.O.dtype, jnp.floating):
        raise ValueError("O.dtype must be floating point number")
    
    if not jnp.issubdtype(hmm.mu.dtype, jnp.floating):
        raise ValueError("mu.dtype must be floating point number")

    # Value assertions that O, T, mu are valid probability distributions
    if not hmm.is_log:
        if not jnp.allclose(jnp.sum(hmm.T, axis=1), 1.0):
            raise ValueError("Rows of T must sum to 1")

        if jnp.any(hmm.T < 0):
            raise ValueError("T must be non-negative")

        if not jnp.allclose(jnp.sum(hmm.O, axis=1), 1.0):
            raise ValueError("Rows of T must sum to 1")

        if jnp.any(hmm.O < 0):
            raise ValueError("T must be non-negative")

        if not jnp.allclose(jnp.sum(hmm.mu, axis=-1), 1.0):
            raise ValueError("mu distributions must all sum to 1")

        if jnp.any(hmm.mu < 0):
            raise ValueError("mu must be non-negative")
        
    if hmm.is_log:
        if not jnp.allclose(logsumexp(hmm.T, axis=1), 0.0):
            raise ValueError("Rows of T must sum to 1 (logsumexp of logprobs must be 0)")

        if not jnp.allclose(logsumexp(hmm.O, axis=1), 0.0):
            raise ValueError("Rows of T must sum to 1 (logsumexp of logprobs must be 0)")

        if not jnp.allclose(logsumexp(hmm.mu, axis=-1), 0.0):
            raise ValueError("mu distributions must all sum to 1 (logsumexp of logprobs must be 0)")

