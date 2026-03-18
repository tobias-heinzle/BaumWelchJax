
import jax.numpy as jnp
from jax import Array

from ..models import HiddenMarkovParameters
from . import wrapped_jit

@wrapped_jit()
def standardize_shapes(obs: Array, hmm: HiddenMarkovParameters) -> tuple[Array, Array]:
    '''Standardize the shapes of `obs` and the initial state distributions `mu`
    to contain a leading axis. 
    
    If multiple observations are emitted per time step, outputs are of shape `obs.shape = (k, l, m)` 
    and `mu.shape = (k, n)`, where `k` is the number of sequences, `l` is the length of the sequences, 
    `m` is the number of observations per time step and `n` is the number of states.

    If only a single observation is output per timestep, the final observation dimension is 
    omitted, i.e. `obs.shape = (k, l)`
    '''

    
    parallel_mode = obs.ndim > (2 if hmm.O.has_multiple_outputs else 1)
    multiple_mu = hmm.mu.ndim > 1

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