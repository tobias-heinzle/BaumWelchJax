from typing import NamedTuple

import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array

from ..util import wrapped_jit, normalize_rows
from ..models import HiddenMarkovModel

class ForwardBackwardResult(NamedTuple):
    '''
    Structured tuple for the results. Contains the fields:

    `gamma`: entries `gamma[i,j]` denote the probabilities of being in state `j` for each time `i`
    
    `xi`: entries`xi[i,j,k]` denote the probabilities of transitioning from state `j` to state `k` at time `i`
    '''
    
    gamma: Array
    xi: Array

@wrapped_jit(static_argnames=['mode'])
def forward_backward(obs: Array, hmm: HiddenMarkovModel, mode: str = 'log') -> ForwardBackwardResult:
    '''
    Computes the forward and backward probability distributions of being in a given state,
    conditioned on all observations prior and after. All in one single pass over the observations.
    The mode parameter can either be 'log' or 'regular' and controls wether computations are carried out
    using the log probabilities or the regular probabilities. Log mode is standard, regular mode will often
    result in numerical underflow and is just present for sanity checking the implementation.
    
    :param obs: Sequence of observations
    :type obs: Array
    :param hmm: Hidden Markov model parameters
    :type hmm: HiddenMarkovModel
    :param mode: Flag to indicate calculations performed in `log` or `regular` space
    :type mode: str
    :return: Resulting conditional distributions `gamma` and `xi`
    :rtype: ForwardBackwardResult
    '''

    if not jnp.issubdtype(obs.dtype, jnp.integer):
        raise ValueError(f'obs must be 1D vector of integers! obs.dtype = {obs.dtype}')

    if mode == 'log':
        if not hmm.is_log:
            hmm = hmm.to_log()
        gamma, xi = _forward_backward_log(obs, hmm)
        
    elif mode == 'regular':
        if hmm.is_log:
            hmm = hmm.to_prob()
        gamma, xi = _forward_backward(obs, hmm)
    else:
        raise ValueError('mode argument must be either "log" or "regular"!')
    
    return ForwardBackwardResult(gamma=gamma, xi=xi)

@wrapped_jit()
def _forward_backward(obs: Array, hmm: HiddenMarkovModel) -> tuple[Array, Array]:

    if hmm.is_log:
        raise ValueError('HiddenMarkovModel (hmm) must be passed in regular probability mode')

    n = hmm.mu.shape[0]
    t_max = len(obs)

    # Initialize forward probabilities
    alpha_0 = hmm.mu * hmm.O[:, obs[0]]
    alpha_0 = normalize_rows(alpha_0)  # alpha_0 / jnp.sum(alpha_0)

    # Initialize backward probabilities
    beta_t_max = jnp.ones(n) / n

    def step(carry, t):
        alpha, beta = carry

        alpha = (alpha @ hmm.T) * hmm.O[:, obs[t]]
        beta = hmm.T @ (hmm.O[:, obs[t_max - t]] * beta)

        alpha = normalize_rows(alpha)  # alpha / jnp.sum(alpha)
        beta = normalize_rows(beta)  # beta / jnp.sum(beta)

        return (alpha, beta), (alpha, beta)

    # Calculate alpha and beta iteratively
    _, (alpha, beta) = lax.scan(
        f=step,
        init=(alpha_0, beta_t_max),
        xs=jnp.arange(1, t_max)
    )

    # Join with the initial values
    alpha = jnp.concat([alpha_0[None, :], alpha])
    beta = jnp.concat([beta_t_max[None, :], beta])

    # Reverse beta
    beta = jnp.flip(beta, axis=0)

    gamma = (alpha * beta)
    gamma = normalize_rows(gamma)

    # Calculation of the xi tensor involves taking the outer product of alpha and O * beta
    # for each combination of alpha_t and beta_t+1
    obs_probs = jnp.take(hmm.O, obs[1:], axis=1).T
    xi = jnp.einsum("ij, ik->ijk", alpha[:-1], beta[1:] * obs_probs)

    # and then multiplying each slice componentwise with
    xi = xi * hmm.T[None, ...]

    xi = xi / jnp.sum(xi, axis=(1, 2))[:, None, None]

    return gamma, xi

@wrapped_jit()
def _forward_backward_log(obs: Array, hmm_log: HiddenMarkovModel) -> tuple[Array, Array]:

    if not hmm_log.is_log:
        raise ValueError('HiddenMarkovModel (hmm_log) must be passed in log mode!')

    n = hmm_log.mu.shape[0]
    t_max = len(obs)

    log_T = hmm_log.T
    log_O = hmm_log.O
    log_mu = hmm_log.mu

    # Initialize forward probabilities
    alpha_0 = log_mu + log_O[:, obs[0]]
    alpha_0 = alpha_0 - logsumexp(alpha_0)

    # Initialize backward probabilities
    beta_t_max = jnp.log(jnp.ones(n) / n)

    def step(carry, t):
        alpha, beta = carry

        alpha = logsumexp(
            alpha[:, None] + log_T, axis=0) + log_O[:, obs[t]]

        beta = logsumexp(
            log_T + (log_O[:, obs[t_max - t]] + beta)[None, :], 
            axis=1)

        alpha = alpha - logsumexp(alpha)
        beta = beta - logsumexp(beta)

        return (alpha, beta), (alpha, beta)

    # Calculate alpha and beta iteratively
    _, (alpha, beta) = lax.scan(
        f=step,
        init=(alpha_0, beta_t_max),
        xs=jnp.arange(1, t_max)
    )

    # Join with the initial values
    alpha = jnp.concat([alpha_0[None, :], alpha])
    beta = jnp.concat([beta_t_max[None, :], beta])

    # Reverse beta
    beta = jnp.flip(beta, axis=0)

    gamma = alpha + beta
    gamma -= logsumexp(gamma, axis=1)[:,None]

    # Calculation of the xi tensor involves taking the outer product of alpha and O * beta
    # for each combination of alpha_t and beta_t+1
    # This calculation changes a little bit in log space, the outer product multiplications
    # become an addition and the normalization a subtraction of the logsumexp
    obs_logprobs = jnp.take(log_O, obs[1:], axis=1).T

    xi = alpha[:-1, :, None] @ jnp.ones((1, n))
    xi += jnp.matrix_transpose(obs_logprobs[:,:,None] @ jnp.ones((1, n)))
    xi += jnp.matrix_transpose(beta[1:, :, None] @ jnp.ones((1, n)))
    xi += log_T[None, ...]

    # Normalize
    xi -= logsumexp(xi, axis=(1,2))[:, None, None]

    return gamma, xi

