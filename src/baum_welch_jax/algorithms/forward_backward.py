from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array

from ..util import wrapped_jit, normalize_rows, standardize_shapes
from ..models import HiddenMarkovParameters

class ForwardBackwardResult(NamedTuple):
    '''
    Structured tuple for the results. Contains the fields:

    `gamma`: entries `gamma[i,j]` denote the probabilities of being in state `j` for each time `i`
    
    `xi`: entries`xi[i,j,k]` denote the probabilities of transitioning from state `j` to state `k` at time `i`
    '''
    
    gamma: Array
    xi: Array

@wrapped_jit(static_argnames=['mode', 'squeeze'])
def forward_backward(
    obs: Array, 
    hmm: HiddenMarkovParameters, 
    mode: str = 'log', 
    squeeze: bool = True) -> ForwardBackwardResult:
    '''
    Computes the forward and backward probability distributions of being in a given state,
    conditioned on all observations prior and after. All in one single pass over the observations.
    The mode parameter can either be 'log' or 'regular' and controls wether computations are carried out
    using the log probabilities or the regular probabilities. Log mode is standard, regular mode will often
    result in numerical underflow and is just present for sanity checking the implementation.
    
    :param obs: Sequence of observations either shape `(n,)` for a single sequence or `(k, n)` for `k` sequences
    :type obs: Array
    :param hmm: Hidden Markov model parameters. If multiple `mu` are provided, their number must match the number of sequences.
    :type hmm: HiddenMarkovModel
    :param mode: Flag to indicate calculations performed in `log` or `regular` space
    :type mode: str
    :param squeeze: If `squeeze` is set to true, the leading axis of the return values will only be kept if it has length > 1.
    :type squeeze: bool
    :return: Resulting conditional distributions `gamma` and `xi`
    :rtype: ForwardBackwardResult
    '''

    if not jnp.issubdtype(obs.dtype, jnp.integer):
        raise ValueError(f'obs must be 1D vector of integers! obs.dtype = {obs.dtype}')

    obs, mu = standardize_shapes(obs, hmm)

    hmm = HiddenMarkovParameters(hmm.T, hmm.O, mu, hmm.is_log)

    # Ensure that the HMM parametes are passed in the correct form
    if mode == 'log':
        if not hmm.is_log:
            hmm = hmm.to_log()
        gamma, xi = jax.vmap(lambda _o, _mu: _forward_backward_log_impl(_o, hmm.T, hmm.O, _mu))(obs, hmm.mu)
        
    elif mode == 'regular':
        if hmm.is_log:
            hmm = hmm.to_prob()
        gamma, xi = jax.vmap(lambda _o, _mu: _forward_backward_impl(_o, hmm.T, hmm.O, _mu))(obs, hmm.mu)
    else:
        raise ValueError('mode argument must be either "log" or "regular"!')
    
    if squeeze:
        return ForwardBackwardResult(gamma=gamma.squeeze(), xi=xi.squeeze())
    else:
        return ForwardBackwardResult(gamma=gamma, xi=xi)
    
@wrapped_jit()
def _forward_backward_impl(obs: Array, T: Array, O: Array, mu: Array) -> tuple[Array, Array]:

    n = mu.shape[0]
    t_max = len(obs)

    # Initialize forward probabilities
    alpha_0 = mu * O[:, obs[0]]
    alpha_0 = normalize_rows(alpha_0) 

    # Initialize backward probabilities
    beta_t_max = jnp.ones(n) / n

    def step(carry, t):
        alpha, beta = carry

        alpha = (alpha @ T) * O[:, obs[t]]
        beta = T @ (O[:, obs[t_max - t]] * beta)

        alpha = normalize_rows(alpha) 
        beta = normalize_rows(beta)

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
    obs_probs = jnp.take(O, obs[1:], axis=1).T
    xi = jnp.einsum("ij, ik->ijk", alpha[:-1], beta[1:] * obs_probs)

    # and then multiplying each slice componentwise with
    xi = xi * T[None, ...]

    xi = xi / jnp.sum(xi, axis=(1, 2))[:, None, None]

    return gamma, xi

@wrapped_jit()
def _forward_backward_log_impl(obs: Array, log_T: Array, log_O: Array, log_mu: Array) -> tuple[Array, Array]:

    n = log_mu.shape[0]
    t_max = len(obs)

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

