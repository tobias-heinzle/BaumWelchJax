from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HiddenMarkovModel:
    T: Array    # Transition (log)probabilities
    O: Array    # Observation (log)probabilities
    mu: Array   # Initial state (log)probabilities

    # Indicates if probabilities are regular or log probs
    is_log: bool = field(metadata={"static": True}, default=False)

    def to_log(self):
        if self.is_log:
            raise ValueError('Only regular probabilities can be transformed to log!')

        return HiddenMarkovModel(
            jnp.log(self.T),
            jnp.log(self.O),
            jnp.log(self.mu),
            is_log=True
        )
    
    def to_prob(self):
        if not self.is_log:
            raise ValueError('Only log probabilities can be transformed to regular!')
            
        return HiddenMarkovModel(
            jnp.exp(self.T),
            jnp.exp(self.O),
            jnp.exp(self.mu),
            is_log=False
        )


def check_valid_hmm(hmm: HiddenMarkovModel) -> bool:
    '''JIT save validation for HiddenMarkovModels.
    Returns a bool that indicates if validation was succesful. '''


    correct_dims = jnp.all(jnp.array([
        hmm.T.ndim == 2, 
        hmm.O.ndim == 2, 
        hmm.mu.ndim == 1, 
        hmm.T.shape[0] == hmm.T.shape[1]
    ]))
    
    if hmm.is_log:
        all_sum_to_one = jnp.all(jnp.array([
            jnp.allclose(logsumexp(hmm.T, axis=1), 0.0),
            jnp.allclose(logsumexp(hmm.O, axis=1), 0.0),
            jnp.allclose(logsumexp(hmm.mu), 0.0)
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
            jnp.allclose(jnp.sum(hmm.mu), 1.0)
        ]))
    
        return jnp.all(jnp.array([correct_dims, all_positive, all_sum_to_one]))

def assert_valid_hmm(hmm: HiddenMarkovModel):
    '''
    Runs assertions for critical properties of a HiddenMarkovModel.
    Throws a ValueError if anything is incorrect.
    '''
    
    # Shape checks for O, T, mu
    if hmm.T.ndim != 2:
        raise ValueError("T must be a 2D probability vector")
    
    if hmm.O.ndim != 2:
        raise ValueError("O must be a 2D probability vector")

    if hmm.mu.ndim != 1:
        raise ValueError("mu must be a 1D probability vector")

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

        if not jnp.allclose(jnp.sum(hmm.mu), 1.0):
            raise ValueError("mu must sum to 1")

        if jnp.any(hmm.mu < 0):
            raise ValueError("mu must be non-negative")
        
    if hmm.is_log:
        if not jnp.allclose(logsumexp(hmm.T, axis=1), 0.0):
            raise ValueError("Rows of T must sum to 1")

        if not jnp.allclose(logsumexp(hmm.O, axis=1), 0.0):
            raise ValueError("Rows of T must sum to 1")

        if not jnp.allclose(logsumexp(hmm.mu), 0.0):
            raise ValueError("mu must sum to 1")

