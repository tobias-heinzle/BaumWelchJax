import jax
import jax.lax as lax
import jax.numpy as jnp

from jax import Array

from logsumexp import logsumexp


@jax.jit
def likelihood(observations: Array, T: Array, O: Array, mu: Array) -> tuple[Array, Array]:
    """
    Compute the likelihood of observing the sequence `obs` given the parameters `T`, `O` and `mu`

    Returns:
        - `state_likelihoods`, the likelihood of being in a given state at the end of a sequence
        - `likelihood_sequence`, where each entry corresponds to the likelihood of the observation
    sequence up to that index. `likelihood_sequence[-1]` is the likelihood of the entire sequence.

    """

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


@jax.jit
def log_likelihood(observations: Array, T: Array, O: Array, mu: Array) -> tuple[Array, Array]:
    """
    Compute the likelihood of observing the sequence `obs` given the parameters `T`, `O` and `mu`
    """

    def loop_body(state_loglikelihoods, obs):
        state_loglikelihoods = jax.vmap(logsumexp)(
            state_loglikelihoods[None, :] + jnp.log(T)) + jnp.log(O[:, obs])
        return state_loglikelihoods, logsumexp(state_loglikelihoods)

    initial_loglikelihoods = jnp.log(mu) + jnp.log(O[:, observations[0]])

    state_loglikelihoods, loglikelihood_sequence = lax.scan(
        loop_body,
        initial_loglikelihoods,
        observations[1:],
        unroll=False
    )

    state_loglikelihoods = jnp.concat(
        [initial_loglikelihoods,
         state_loglikelihoods]
    )

    loglikelihood_sequence = jnp.concat(
        [jnp.sum(initial_loglikelihoods)[None],
         loglikelihood_sequence]
    )

    return state_loglikelihoods, loglikelihood_sequence
