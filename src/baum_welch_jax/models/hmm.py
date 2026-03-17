from dataclasses import dataclass, field
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from .observation_models import ObservationModel

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

    T: Array            # Transition (log)probabilities
    O: ObservationModel # Observation model
    mu: Array           # Initial state (log)probabilities

    # Indicates if probabilities are regular or log probs
    is_log: bool = field(metadata={"static": True}, default=False)

    def to_log(self) -> Self:
        if self.is_log:
            raise ValueError('Only regular probabilities can be transformed to log!')

        return HiddenMarkovParameters(
            jnp.log(self.T),
            self.O.to_log(),
            jnp.log(self.mu),
            is_log=True
        )
    
    def to_prob(self) -> Self:
        if not self.is_log:
            raise ValueError('Only log probabilities can be transformed to regular!')
            
        return HiddenMarkovParameters(
            jnp.exp(self.T),
            self.O.to_prob(),
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
    
    def __str__(self):
        return f'''T =\n{self.T}\n\nO =\n{self.O}\n\nmu=\n{self.mu}'''

    @property
    def is_valid(self) -> bool:
        '''JIT save validation for HiddenMarkovModels.
        Returns a bool that indicates if validation was succesful. '''

        valid_obs_model = self.O.is_valid

        correct_dims = jnp.all(jnp.array([
            self.T.ndim == 2, 
            # hmm.O.ndim == 2, 
            self.mu.ndim == 1 or self.mu.ndim == 2, 
            self.T.shape[0] == self.T.shape[1]
        ]))

        is_float = (
            jnp.issubdtype(self.T.dtype, jnp.floating) 
            # & jnp.issubdtype(hmm.O.dtype, jnp.floating) 
            & jnp.issubdtype(self.mu.dtype, jnp.floating)
        )
        
        if self.is_log:
            all_sum_to_one = jnp.all(jnp.array([
                jnp.allclose(logsumexp(self.T, axis=1), 0.0),
                # jnp.allclose(logsumexp(hmm.O, axis=1), 0.0),
                jnp.allclose(logsumexp(self.mu, axis=-1), 0.0)
            ]))

            return jnp.all(jnp.array([valid_obs_model, correct_dims, all_sum_to_one]))

        else:
            all_positive = jnp.all(jnp.array([
                jnp.all(self.T >= 0),
                # jnp.all(hmm.O >= 0),
                jnp.all(self.mu >= 0)
            ]))

            all_sum_to_one = jnp.all(jnp.array([
                jnp.allclose(jnp.sum(self.T, axis=1), 1.0),
                # jnp.allclose(jnp.sum(hmm.O, axis=1), 1.0),
                jnp.allclose(jnp.sum(self.mu, axis=-1), 1.0)
            ]))
        
            return jnp.all(jnp.array([valid_obs_model, correct_dims, all_positive, all_sum_to_one, is_float]))
        
    # TODO: Write a test for this
    def construct_frozen_parameter_pytree(self, freeze_masks: FreezeMasks) -> HiddenMarkovParameters:
        '''Based on a set of freeze masks, construct a pytree that can be used in a tree map
        to perform a masked parameter update.'''
        O_mask = self.O.construct_frozen_parameter_pytree(freeze_masks.O)

        return HiddenMarkovParameters(freeze_masks.T, O_mask, freeze_masks.mu, self.is_log)


def assert_valid_hmm(hmm: HiddenMarkovParameters):
    '''
    Runs assertions for critical properties of a HiddenMarkovModel.
    Throws a `ValueError` if anything is incorrect.
    '''

    if not hmm.O.is_valid:
        raise ValueError('Observation model is invalid')

    # Shape checks for O, T, mu
    if hmm.T.ndim != 2:
        raise ValueError("T must be a 2D matrix")
    
    if hmm.mu.ndim > 2 or hmm.mu.ndim < 1:
        raise ValueError("mu must be a either a 1D or 2D array")
    
    if not jnp.issubdtype(hmm.T.dtype, jnp.floating):
        raise ValueError("T.dtype must be floating point number")
    
    if not jnp.issubdtype(hmm.mu.dtype, jnp.floating):
        raise ValueError("mu.dtype must be floating point number")

    # Value assertions that O, T, mu are valid probability distributions
    if not hmm.is_log:
        if not jnp.allclose(jnp.sum(hmm.T, axis=1), 1.0):
            raise ValueError("Rows of T must sum to 1")

        if jnp.any(hmm.T < 0):
            raise ValueError("T must be non-negative")

        if not jnp.allclose(jnp.sum(hmm.mu, axis=-1), 1.0):
            raise ValueError("mu distributions must all sum to 1")

        if jnp.any(hmm.mu < 0):
            raise ValueError("mu must be non-negative")
        
    if hmm.is_log:
        if not jnp.allclose(logsumexp(hmm.T, axis=1), 0.0):
            raise ValueError("Rows of T must sum to 1 (logsumexp of logprobs must be 0)")

        if not jnp.allclose(logsumexp(hmm.mu, axis=-1), 0.0):
            raise ValueError("mu distributions must all sum to 1 (logsumexp of logprobs must be 0)")


# TODO: Refactor the parameter freezing to handle general observation models!

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FreezeMasks:
    """Arrays indicating which HMM parameters are held fixed during inference."""
    T: Array
    O: Array
    mu: Array


@dataclass(frozen=True)
class FreezeConfig:
    """Flags indicating which HMM parameter arrays are held fixed during inference."""
    T: bool = False
    O: bool = False
    mu: bool = False

    def create_masks(self, hmm: HiddenMarkovParameters) -> FreezeMasks:
        return FreezeMasks(
            T=jnp.full_like(hmm.T, self.T, dtype=jnp.bool),
            O=jnp.full_like(hmm.O.get_params(), self.O, dtype=jnp.bool),
            mu=jnp.full_like(hmm.mu, self.mu, dtype=jnp.bool),
        )
