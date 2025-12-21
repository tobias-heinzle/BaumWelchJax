
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array

from ..util import wrapped_jit

@wrapped_jit(static_argnames=["return_stats"])
def likelihood(obs: Array, T: Array, O: Array, mu: Array, return_stats: bool = False) -> Array | tuple[Array, Array]:
    """
    Compute the likelihood of observing the sequence `obs` given the parameters `T`, `O` and `mu`

    Returns:
        - Likelihood of the sequence.

        If `return_stats = True` instead you get:

        - `state_likelihoods`, the likelihood of being in a given state at the end of a sequence
        - `likelihood_sequence`, where each entry corresponds to the likelihood of the observation
    sequence up to that index. `likelihood_sequence[-1]` is the likelihood of the entire sequence.

    """

    def loop_body(llhood, obs):
        llhood = llhood @ T * O[:, obs]
        return llhood, jnp.sum(llhood)

    initial_likelihoods = mu * O[:, obs[0]]


    state_likelihoods, sequence_likelihoods = lax.scan(
        loop_body,
        initial_likelihoods,
        obs[1:],
        unroll=False
    )

    sequence_likelihoods = jnp.concat(
        [jnp.sum(initial_likelihoods)[None],
         sequence_likelihoods]
    )

    if return_stats:
        return state_likelihoods, sequence_likelihoods
    else:
        return sequence_likelihoods[-1]


@wrapped_jit(static_argnames=["return_stats"])
def log_likelihood(obs: Array, T: Array, O: Array, mu: Array, return_stats: bool = False) -> Array | tuple[Array, Array]:
    """
    Compute the likelihood of observing the sequence `obs` given the parameters `T`, `O` and `mu`

    Returns:
        - Log likelihood of the sequence.

        If `return_stats = True` instead you get:

        - `state_loglikelihoods`, the log likelihood of being in a given state at the end of a sequence
        - `loglikelihood_sequence`, where each entry corresponds to the log likelihood of the observation
    sequence up to that index. `loglikelihood_sequence[-1]` is the log likelihood of the entire sequence.

    """

    log_T = jnp.log(T)
    log_O = jnp.log(O)

    def loop_body(log_llhood, obs):
        log_llhood = logsumexp(
            log_llhood[:, None] + log_T, axis=0) + log_O[:, obs]
        return log_llhood, logsumexp(log_llhood)

    initial_loglikelihoods = jnp.log(mu) + log_O[:, obs[0]]

    state_loglikelihoods, loglikelihood_sequence = lax.scan(
        loop_body,
        initial_loglikelihoods,
        obs[1:],
        unroll=False
    )

    loglikelihood_sequence = jnp.concat(
        [logsumexp(initial_loglikelihoods)[None],
         loglikelihood_sequence]
    )

    if return_stats:
        return state_loglikelihoods, loglikelihood_sequence
    else:
        return loglikelihood_sequence[-1]
