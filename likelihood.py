import jax
import jax.lax as lax
import jax.numpy as jnp

from jax import Array


@jax.jit
def likelihood(observations: Array, T: Array, O: Array, mu: Array) -> tuple[Array, Array]:
    # Compute the likelihood of observing the sequence obs given the parameters T, O and mu

    def loop_body(state_likelihoods, obs):
        state_likelihoods = state_likelihoods @ T * O[:, obs]
        return state_likelihoods, jnp.sum(state_likelihoods)

    initial_likelihoods = mu * O[:, observations[0]]

    state_likelihoods, likelihood_sequence = lax.scan(
        loop_body,
        initial_likelihoods,
        observations[1:],
        unroll=False
    )

    state_likelihoods = jnp.concat(
        [initial_likelihoods,
         state_likelihoods]
    )

    likelihood_sequence = jnp.concat(
        [jnp.sum(initial_likelihoods)[None],
         likelihood_sequence]
    )

    return state_likelihoods, likelihood_sequence
