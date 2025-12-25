
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array

from ..util import wrapped_jit
from ..models import HiddenMarkovModel

@wrapped_jit(static_argnames=["return_stats"])
def likelihood(obs: Array, hmm: HiddenMarkovModel, return_stats: bool = False) -> Array | tuple[Array, Array]:
    """
    Compute the likelihood of observing the sequence `obs` given the parameters of `hmm`

    Returns:
        - Likelihood of the sequence.

        If `return_stats = True` instead you get:

        - `state_likelihoods`, the likelihood of being in a given state at the end of a sequence
        - `likelihood_sequence`, where each entry corresponds to the likelihood of the observation
    sequence up to that index. `likelihood_sequence[-1]` is the likelihood of the entire sequence.

    """
    if not jnp.issubdtype(obs.dtype, jnp.integer):
        raise ValueError(f'obs must be 1D vector of integers! obs.dtype = {obs.dtype}')

    if hmm.is_log:
        hmm = hmm.to_prob()

    def loop_body(llhood, obs):
        llhood = (llhood @ hmm.T) * hmm.O[:, obs]
        return llhood, jnp.sum(llhood)

    initial_likelihoods = hmm.mu * hmm.O[:, obs[0]]


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
def log_likelihood(obs: Array, hmm: HiddenMarkovModel, return_stats: bool = False) -> Array | tuple[Array, Array]:
    """
    Compute the likelihood of observing the sequence `obs` given the parameters of `hmm`

    Returns:
        - Log likelihood of the sequence.

        If `return_stats = True` instead you get:

        - `state_loglikelihoods`, the log likelihood of being in a given state at the end of a sequence
        - `loglikelihood_sequence`, where each entry corresponds to the log likelihood of the observation
    sequence up to that index. `loglikelihood_sequence[-1]` is the log likelihood of the entire sequence.

    """
    if not jnp.issubdtype(obs.dtype, jnp.integer):
        raise ValueError(f'obs must be 1D vector of integers! obs.dtype = {obs.dtype}')

    if not hmm.is_log:
        hmm = hmm.to_log()

    log_T = hmm.T
    log_O = hmm.O
    log_mu = hmm.mu


    def loop_body(log_llhood, obs):
        log_llhood = logsumexp(
            log_llhood[:, None] + log_T, axis=0) + log_O[:, obs]
        return log_llhood, logsumexp(log_llhood)

    initial_loglikelihoods = log_mu + log_O[:, obs[0]]

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
