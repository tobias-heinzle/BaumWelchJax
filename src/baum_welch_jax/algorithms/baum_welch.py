from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array, vmap

from ..util import wrapped_jit, normalize_rows
from ..models import HiddenMarkovModel
from .forward_backward import forward_backward
from .likelihoods import log_likelihood


# TODO: How to deal with sequences that have different initial states mu
#       How to handle ragged seqeunces of different length?

class IterationState(NamedTuple):
    params: HiddenMarkovModel
    log_likelihoods: Array
    residuals: Array
    iterations: int
    terminated: bool

def _require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "baum_welch requires JAX double precision (jax_enable_x64=True).\n"
            "Enable it when importing JAX:\n\n"
            "  import jax\n"
            "  jax.config.update('jax_enable_x64', True)\n"
        )    

def _maximization_step(obs: Array, gamma: Array, xi: Array, m: int) -> HiddenMarkovModel:

    # Average over all time steps and normalize along rows => new estimate for T

    T = jnp.sum(xi, axis=0)
    T = normalize_rows(T)

    O = lax.map(lambda o: jnp.sum(
        (obs.ravel() == o)[:, None] * gamma, axis=0), jnp.arange(m)).T
    O = O / jnp.sum(gamma, axis=0)[..., None]

    mu = gamma[0]

    return HiddenMarkovModel(T, O, mu)


def _maximization_step_log(obs: Array, gamma: Array, xi: Array, m: int) -> HiddenMarkovModel:
    
    # Average over all time steps and normalize along rows => new estimate for T

    T = logsumexp(xi, axis=0)
    T -= logsumexp(T, axis=-1)[..., None]

    O = lax.map(lambda o: logsumexp(
        jnp.log(obs.ravel() == o)[:, None] + gamma, axis=0), jnp.arange(m)).T
    O -= logsumexp(gamma, axis=0)[..., None]

    mu = gamma[0] # - logsumexp(gamma[0])

    return HiddenMarkovModel(T, O, mu, is_log=True)


def _compute_residual(updated: HiddenMarkovModel, old: HiddenMarkovModel, mode: str = 'log') -> float:
    if mode == 'log':
        residual_T = jnp.max(jnp.abs(jnp.exp(updated.T) - jnp.exp(old.T)))
        residual_O = jnp.max(jnp.abs(jnp.exp(updated.O) - jnp.exp(old.O)))
        residual_mu = jnp.max(jnp.abs(jnp.exp(updated.mu) - jnp.exp(old.mu)))
    elif mode == 'regular':
        residual_T = jnp.max(jnp.abs(updated.T - old.T))
        residual_O = jnp.max(jnp.abs(updated.O - old.O))
        residual_mu = jnp.max(jnp.abs(updated.mu - old.mu))
    else:
        raise ValueError('`mode` must be either `log` or `regular`')
    
    return jnp.max(jnp.array([residual_T, residual_O, residual_mu]))
    

@wrapped_jit(static_argnames=["max_iter", "epsilon", "mode"])
def baum_welch(obs: Array,
        initial_params: HiddenMarkovModel,
        max_iter=100,
        epsilon=1e-4,
        mode='log') -> IterationState:
    
    _require_x64()
    
    if mode == 'log':
        if not initial_params.is_log:
            initial_params = initial_params.to_log()
        
    elif mode == 'regular':
        if initial_params.is_log:
            initial_params = initial_params.to_prob()
    else:
        raise ValueError('mode argument must be either "log" or "regular"!')
    
    parallel_mode = len(obs.shape) > 1

    m = initial_params.O.shape[-1]

    def iteration(
            carry: IterationState,
            _: None
            ) -> tuple[IterationState, None]:


        def perform_step(
                inner_carry: IterationState
                ) -> IterationState:

            # Expectation - step
            # (Forward-Backward algorithm for estimating probabilities 
            # of the latent variables (states) given the observations)
            if parallel_mode:
                gamma, xi = vmap(
                    lambda _o: forward_backward(_o, inner_carry.params, mode=mode))(obs)
                
                _log_llhood = vmap(lambda _o: log_likelihood(_o, inner_carry.params))(obs)
                log_llhood = jnp.sum(_log_llhood)

                gamma = jnp.concat(gamma, axis=0)
                xi = jnp.concat(xi, axis=0)
            else:
                gamma, xi = forward_backward(obs, inner_carry.params, mode=mode)

                log_llhood = log_likelihood(obs, inner_carry.params)

            # Maximizaton - step
            if mode == 'log':
                updated = _maximization_step_log(obs, gamma, xi, m)
            else:
                updated = _maximization_step(obs, gamma, xi, m)

            residual = _compute_residual(updated, inner_carry.params, mode=mode)    

            residuals = inner_carry.residuals.at[inner_carry.iterations].set(residual)
            log_llhoods = inner_carry.log_likelihoods.at[inner_carry.iterations].set(log_llhood)

            return lax.cond(
                jnp.any(jnp.isnan(updated.T)) | jnp.any(jnp.isnan(updated.O) | jnp.any(jnp.isnan(updated.mu))),
                lambda: IterationState(updated, log_llhoods, residuals, inner_carry.iterations, True),
                lambda: IterationState(updated, log_llhoods, residuals, inner_carry.iterations + 1, False)

            )

        hmm, log_llhoods, residuals, n_step, terminated = carry

        # The log likelihood difference of two subsequent iterations is used to determine convergence
        log_llhood_diff = log_llhoods[n_step - 1] - log_llhoods[n_step - 2]

        carry = lax.cond(
            # Terminate if any NaN values are encountered or the log_likelihoods have stopped increasing
            terminated | ((n_step >= 2) & (log_llhood_diff < epsilon)),
            lambda x: x,
            perform_step,
            IterationState(hmm, log_llhoods, residuals, n_step, terminated)

        )

        return carry, None

    final_state, _ = lax.scan(
        iteration,
        init=IterationState(
            params=initial_params, 
            log_likelihoods=jnp.ones(max_iter) * (- jnp.inf), 
            residuals=jnp.ones(max_iter) * jnp.inf, 
            iterations=0, 
            terminated=False),
        length=max_iter
    )
    
    return final_state