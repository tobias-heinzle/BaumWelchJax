
import jax
import jax.lax as lax
import jax.numpy as jnp

from jax import Array

from ..util import wrapped_jit
from ..models import HiddenMarkovParameters


@wrapped_jit(static_argnames="length")
def generate_sequence(
        key: Array,
        hmm: HiddenMarkovParameters, 
        length: int) -> tuple[Array, Array]:
    '''
    Generate a sequence of states and observations with a given length for each initial state distribution
    in the HiddenMarkovParameters
    
    :param key: RNG key for sequence generation
    :type key: Array
    :param hmm: Hidden Markov model parameters
    :type hmm: HiddenMarkovModel
    :param length: Length of the output sequence
    :type length: int
    :return: Two arrays containing `states` and `observations`
    :rtype: tuple[Array, Array]
    '''

    if hmm.is_log:
        hmm = hmm.to_prob()

    if len(hmm.mu.shape) > 1:
        keys = jnp.array(jax.random.split(key, hmm.mu.shape[0]))
        states, observations = jax.vmap(
            lambda _k, _mu: _generate_sequence_impl(_k, hmm.T, hmm.O, _mu, length))(keys, hmm.mu)
    else:
        states, observations = _generate_sequence_impl(key, hmm.T, hmm.O, hmm.mu, length)

    return states.squeeze(), observations.squeeze()

@wrapped_jit(static_argnames="length")
def _generate_sequence_impl(key: Array, T: Array, O: Array, mu: Array, length: int):
    n, _ = O.shape

    initial_key, sampling_key = jax.random.split(key)
    initial_state = jax.random.choice(initial_key, n, p=mu)

    p_samples = jax.random.uniform(sampling_key, (length, 2))

    obs_cdf = jnp.cumsum(O, axis=-1)
    trans_cdf = jnp.cumsum(T, axis=-1)

    def step(state, p_samples):
        p_obs, p_state = p_samples

        observation = jnp.argmax(obs_cdf[state] >= p_obs)
        next_state = jnp.argmax(trans_cdf[state] >= p_state)

        return next_state, (state, observation)
    
    _, (states, observations) = lax.scan(step, initial_state, p_samples)

    return states, observations

