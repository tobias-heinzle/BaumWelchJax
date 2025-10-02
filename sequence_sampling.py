from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax.random import uniform, choice, split

from jax import Array


@partial(jax.jit, static_argnames="length")
def generate_sequence(
        key: Array,
        transition_matrix: Array,
        observation_matrix: Array,
        initial_distribution: Array,
        length: int) -> Array:

    n, _ = observation_matrix.shape

    initial_key, sampling_key = split(key)
    initial_state = choice(initial_key, n, p=initial_distribution)

    p_samples = uniform(sampling_key, (length, 2))

    obs_cdf = jnp.cumsum(observation_matrix, axis=-1)
    trans_cdf = jnp.cumsum(transition_matrix, axis=-1)

    def step(state, p_samples):
        p_obs, p_state = p_samples

        observation = jnp.argmax(obs_cdf[state] >= p_obs)
        next_state = jnp.argmax(trans_cdf[state] >= p_state)

        return next_state, (state, observation)

    _, (states, observations) = lax.scan(step, initial_state, p_samples)

    return states, observations
