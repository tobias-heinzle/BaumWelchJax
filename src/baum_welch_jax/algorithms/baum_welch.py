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

        return _baum_welch_log(obs, initial_params, max_iter, epsilon)
    elif mode == 'regular':
        if initial_params.is_log:
            initial_params = initial_params.to_prob()

        return _baum_welch(obs, initial_params, max_iter, epsilon)
    else:
        raise ValueError('mode argument must be either "log" or "regular"!')
    

@wrapped_jit(static_argnames=["max_iter", "epsilon"])
def _baum_welch(
        obs: Array,
        initial_params: HiddenMarkovModel,
        max_iter=100,
        epsilon=1e-4) -> IterationState:

    if initial_params.is_log:
        raise ValueError('initial_params must be probabilities!')
    
    parallel_mode = len(obs.shape) > 1

    m = initial_params.O.shape[-1]

    def iteration(carry: IterationState, _: None) -> tuple[IterationState, None]:

        def perform_step(
                inner_carry: IterationState
                ) -> IterationState:

            # Expectation - step
            # (Forward-Backward algorithm for estimating probabilities 
            # of the latent variables (states) given the observations)
            if parallel_mode:
                gamma, xi = vmap(
                    lambda _o: forward_backward(_o, inner_carry.params, mode='regular'))(obs)
                _log_llhood = vmap(lambda _o: log_likelihood(_o, inner_carry.params))(obs)
                log_llhood = jnp.sum(_log_llhood)

                gamma = jnp.concat(gamma, axis=0)
                xi = jnp.concat(xi, axis=0)
            else:
                gamma, xi = forward_backward(obs, inner_carry.params, mode='regular')
                log_llhood = log_likelihood(obs, inner_carry.params)

            # Maximization - step
            # Average over all time steps and normalize along rows => new estimate for T
            T = jnp.sum(xi, axis=0)
            T = normalize_rows(T)  # T / jnp.sum(T, axis=-1)[..., None]

            O = lax.map(lambda o: jnp.sum(
                (obs.ravel() == o)[:, None] * gamma, axis=0), jnp.arange(m)).T
            O = O / jnp.sum(gamma, axis=0)[..., None]

            mu = gamma[0]

            residual_T = jnp.max(jnp.abs(T - inner_carry.params.T))
            residual_O = jnp.max(jnp.abs(O - inner_carry.params.O))
            residual_mu = jnp.max(jnp.abs(mu - inner_carry.params.mu))

            residual = jnp.max(jnp.array([residual_T, residual_O, residual_mu]))

            updated_hmm = HiddenMarkovModel(T, O, mu)

            residuals = inner_carry.residuals.at[inner_carry.iterations].set(residual)
            log_llhoods = inner_carry.log_likelihoods.at[inner_carry.iterations].set(log_llhood)

            return lax.cond(
                jnp.any(jnp.isnan(T)) | jnp.any(jnp.isnan(O) | jnp.any(jnp.isnan(mu))),
                lambda: IterationState(updated_hmm, log_llhoods, residuals, inner_carry.iterations, True),
                lambda: IterationState(updated_hmm, log_llhoods, residuals, inner_carry.iterations + 1, False)
            )
        
        hmm, log_llhoods, residuals, n_step, terminated = carry

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


@wrapped_jit(static_argnames=["max_iter", "epsilon"])
def _baum_welch_log(
        obs: Array,
        initial_params: HiddenMarkovModel,
        max_iter=100,
        epsilon=1e-4) -> IterationState:
    
    if not initial_params.is_log:
        raise ValueError('initial_params must be log probabilities!')

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
                    lambda _o: forward_backward(_o, inner_carry.params, mode='log'))(obs)
                _log_llhood = vmap(lambda _o: log_likelihood(_o, inner_carry.params))(obs)
                log_llhood = jnp.sum(_log_llhood)

                gamma = jnp.concat(gamma, axis=0)
                xi = jnp.concat(xi, axis=0)
            else:
                gamma, xi = forward_backward(obs, inner_carry.params, mode='log')
                log_llhood = log_likelihood(obs, inner_carry.params)

            # Maximizaton - step
            # Average over all time steps and normalize along rows => new estimate for T
            T = logsumexp(xi, axis=0)
            T -= logsumexp(T, axis=-1)[..., None]

            O = lax.map(lambda o: logsumexp(
                jnp.log(obs.ravel() == o)[:, None] + gamma, axis=0), jnp.arange(m)).T
            O -= logsumexp(gamma, axis=0)[..., None]

            mu = gamma[0] - logsumexp(gamma[0])

            residual_T = jnp.max(jnp.abs(jnp.exp(T) - jnp.exp(inner_carry.params.T)))
            residual_O = jnp.max(jnp.abs(jnp.exp(O) - jnp.exp(inner_carry.params.O)))
            residual_mu = jnp.max(jnp.abs(jnp.exp(mu) - jnp.exp(inner_carry.params.mu)))

            residual = jnp.max(jnp.array([residual_T, residual_O, residual_mu]))

            updated_hmm = HiddenMarkovModel(T, O, mu, is_log=True)

            residuals = inner_carry.residuals.at[inner_carry.iterations].set(residual)
            log_llhoods = inner_carry.log_likelihoods.at[inner_carry.iterations].set(log_llhood)

            return lax.cond(
                jnp.any(jnp.isnan(T)) | jnp.any(jnp.isnan(O) | jnp.any(jnp.isnan(mu))),
                lambda: IterationState(updated_hmm, log_llhoods, residuals, inner_carry.iterations, True),
                lambda: IterationState(updated_hmm, log_llhoods, residuals, inner_carry.iterations + 1, False)

            )

        hmm, log_llhoods, residuals, n_step, terminated = carry

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