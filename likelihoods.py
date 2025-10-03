from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax import Array

from logsumexp import logsumexp


@partial(jax.jit, static_argnames=["return_stats"])
def likelihood(observations: Array, T: Array, O: Array, mu: Array, return_stats: bool = False) -> Array | tuple[Array, Array]:
    """
    Compute the likelihood of observing the sequence `obs` given the parameters `T`, `O` and `mu`

    Returns:
        - Likelihood of the sequence.

        If `return_stats = True` instead you get:

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

    if return_stats:
        return state_likelihoods, likelihood_sequence
    else:
        return likelihood_sequence[-1]


@partial(jax.jit, static_argnames=["return_stats"])
def log_likelihood(observations: Array, T: Array, O: Array, mu: Array, return_stats: bool = False) -> Array | tuple[Array, Array]:
    """
    Compute the likelihood of observing the sequence `obs` given the parameters `T`, `O` and `mu`

    Returns:
        - Log likelihood of the sequence.

        If `return_stats = True` instead you get:

        - `state_loglikelihoods`, the log likelihood of being in a given state at the end of a sequence
        - `loglikelihood_sequence`, where each entry corresponds to the log likelihood of the observation
    sequence up to that index. `loglikelihood_sequence[-1]` is the log likelihood of the entire sequence.

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

    if return_stats:
        return state_loglikelihoods, loglikelihood_sequence
    else:
        return loglikelihood_sequence[-1]
