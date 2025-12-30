
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array

from ..util import wrapped_jit, standardize_shapes
from ..models import HiddenMarkovParameters

@wrapped_jit(static_argnames=["return_stats"])
def likelihood(obs: Array, hmm: HiddenMarkovParameters, return_stats: bool = False) -> Array | tuple[Array, Array]:
    '''
    Compute the likelihoods of observing the sequences `obs` given the parameters of `hmm`. This function can return
    two different types:

    If `return_stats = False`
    - Likelihood(s) of the sequence(s) 

    If `return_stats = True`
    - `state_likelihoods` (likelihoods of being in a given state at the end of a sequence)
    - `likelihood_sequence` (likelihoods of the observation sequences at each step)
    
    If multiple sequences are passed, outputs are computed separately for each sequence.
    If additionally also different initial state probabilities mu are passed for each
    sequence, the likelihoods are computed with respect to these initial state probabilities.
    
    :param obs: Observation sequence
    :type obs: Array
    :param hmm: Hidden Markov model parameters
    :type hmm: HiddenMarkovModel
    :param return_stats: Flag to indicate if additional statistics should be returned
    :type return_stats: bool
    :return: Likelihood value(s) or state likelihoods and likelihood sequence
    :rtype: Array | tuple[Array, Array]
    '''
    
    if not jnp.issubdtype(obs.dtype, jnp.integer):
        raise ValueError(f'obs must be of dtype integer! obs.dtype = {obs.dtype}')

    if hmm.is_log:
        hmm = hmm.to_prob()

    obs, mu = standardize_shapes(obs, hmm)

    state_llhoods, llhood_seq = jax.vmap(
        lambda _o, _mu: _likelihood_impl(_o, hmm.T, hmm.O, _mu))(obs, mu)

    if return_stats:
        return state_llhoods.squeeze(), llhood_seq.squeeze()
    else:
        return llhood_seq.squeeze()[-1]

@wrapped_jit()
def _likelihood_impl(obs: Array, T: Array, O: Array, mu: Array) -> tuple[Array, Array]:

    def loop_body(llhood, obs):
        llhood = (llhood @ T) * O[:, obs]
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

    return state_likelihoods, sequence_likelihoods



@wrapped_jit(static_argnames=["return_stats"])
def log_likelihood(obs: Array, hmm: HiddenMarkovParameters, return_stats: bool = False) -> Array | tuple[Array, Array]:
    '''
    Compute the log likelihoods of observing the sequences `obs` given the parameters of `hmm`. This function can return
    two different types:

    If `return_stats = False`
    - Log likelihood(s) of the sequence(s) 

    If `return_stats = True`
    - `state_loglikelihoods` (loglikelihoods of being in a given state at the end of a sequence)
    - `loglikelihood_sequence` (loglikelihoods of the observation sequence up to that index)
    
    If multiple sequences are passed, outputs are computed separately for each sequence.
    If additionally also different initial state probabilities mu are passed for each
    sequence, the log likelihoods are computed with respect to these initial state probabilities.
    
    :param obs: Observation sequence
    :type obs: Array
    :param hmm: Hidden Markov model parameters
    :type hmm: HiddenMarkovModel
    :param return_stats: Flag to indicate if additional statistics should be returned
    :type return_stats: bool
    :return: Log likelihood value(s) or state log likelihoods and log likelihood sequence
    :rtype: Array | tuple[Array, Array]
    '''
    if not jnp.issubdtype(obs.dtype, jnp.integer):
        raise ValueError(f'obs must be 1D vector of integers! obs.dtype = {obs.dtype}')

    if not hmm.is_log:
        hmm = hmm.to_log()

    log_T = hmm.T
    log_O = hmm.O
    
    obs, log_mu = standardize_shapes(obs, hmm)

    state_logllhoods, logllhood_seq = jax.vmap(
        lambda _o, _mu: _log_likelihood_impl(_o, log_T, log_O, _mu))(obs, log_mu)

    if return_stats:
        return state_logllhoods.squeeze(), logllhood_seq.squeeze()
    else:
        return logllhood_seq.squeeze()[-1]

@wrapped_jit()
def _log_likelihood_impl(obs: Array, log_T: Array, log_O: Array, log_mu: Array) -> tuple[Array, Array]:

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

    return state_loglikelihoods, loglikelihood_sequence

