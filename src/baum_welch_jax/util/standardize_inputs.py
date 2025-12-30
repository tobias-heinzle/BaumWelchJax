
import jax.numpy as jnp
from jax import Array

from ..models import HiddenMarkovParameters
from . import wrapped_jit

@wrapped_jit()
def standardize_shapes(obs: Array, hmm: HiddenMarkovParameters) -> tuple[Array, Array]:
    '''Standardize the shapes of `obs` and the initial state distributions `mu`
    to contain a leading axis. Outputs are of shape `(k, l)` and `(k, n)`, where `k` is the
    number of sequences, `l` is the length of the sequences and `n` is the number of states.'''
    
    parallel_mode = len(obs.shape) > 1
    multiple_mu = len(hmm.mu.shape) > 1

    if multiple_mu and (not parallel_mode):
        raise ValueError('Multiple mu distributions provided, but only a single obs sequence!')
    
    if multiple_mu and parallel_mode:
        if len(hmm.mu) != len(obs):
            raise ValueError(
                'If multiple mu distributions are provided, their number must ' 
                'match the number of observation sequences: len(initial_params.mu) != len(obs) '
                f'({len(hmm.mu)} !=  {len(obs)})'
                )
        mu = hmm.mu

    # Ensure that the shape of obs and hmm.mu always has a leading axis over the number of sequences
    if not parallel_mode:
        obs = obs[None, ...]

    if not multiple_mu:
        mu = jnp.repeat(hmm.mu[None, ...], repeats = len(obs), axis=0)

    

    return obs, mu