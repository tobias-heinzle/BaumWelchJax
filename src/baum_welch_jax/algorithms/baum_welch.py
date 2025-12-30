from typing import NamedTuple, Self

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array, vmap

from ..util import wrapped_jit, normalize_rows, standardize_shapes
from ..models import HiddenMarkovParameters
from .forward_backward import forward_backward
from .likelihoods import log_likelihood


# TODO: How to deal with sequences that have different initial states mu
#       How to handle ragged seqeunces of different length?

class IterationState(NamedTuple):
    '''
    Structured tuple for the (intermediate) results of expectation maximization
    '''
    params: HiddenMarkovParameters
    log_likelihoods: Array
    residuals: Array
    iterations: int
    terminated: bool

    def squeeze(self) -> Self:
        return IterationState(
            params=HiddenMarkovParameters(
                self.params.T.squeeze(),
                self.params.O.squeeze(),
                self.params.mu.squeeze(),
                self.params.is_log
            ),
            log_likelihoods=self.log_likelihoods.squeeze(),
            residuals=self.residuals.squeeze(),
            iterations=self.iterations,
            terminated=self.terminated
        )
    
    def replace_mu(self, new_mu: Array) -> Self:
        return IterationState(
            params=HiddenMarkovParameters(
                self.params.T,
                self.params.O,
                new_mu,
                self.params.is_log
            ),
            log_likelihoods=self.log_likelihoods,
            residuals=self.residuals,
            iterations=self.iterations,
            terminated=self.terminated
        )

def _require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "baum_welch requires JAX double precision (jax_enable_x64=True).\n"
            "Enable it when importing JAX:\n\n"
            "  import jax\n"
            "  jax.config.update('jax_enable_x64', True)\n"
        )    

def _maximization_step(obs: Array, gamma: Array, xi: Array, m: int) -> tuple[Array, Array]:

    # Average over all time steps and normalize along rows => new estimate for T

    T = jnp.sum(xi, axis=0)
    T = normalize_rows(T)

    O = lax.map(lambda o: jnp.sum(
        (obs.ravel() == o)[:, None] * gamma, axis=0), jnp.arange(m)).T
    O = O / jnp.sum(gamma, axis=0)[..., None]

    return T, O


def _maximization_step_log(obs: Array, gamma: Array, xi: Array, m: int) -> tuple[Array, Array]:
    
    # Average over all time steps and normalize along rows => new estimate for T

    T = logsumexp(xi, axis=0)
    T -= logsumexp(T, axis=-1)[..., None]

    O = lax.map(lambda o: logsumexp(
        jnp.log(obs.ravel() == o)[:, None] + gamma, axis=0), jnp.arange(m)).T
    O -= logsumexp(gamma, axis=0)[..., None]

    return T, O


def _compute_residual(updated: HiddenMarkovParameters, old: HiddenMarkovParameters, mode: str = 'log') -> float:
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
        initial_params: HiddenMarkovParameters,
        max_iter: int = 100,
        epsilon: float = 1e-6,
        mode: str = 'log') -> IterationState:
    '''
    Implementation of expectation maximization for hidden Markov models.
    `baum_welch` can only be used with x64 precision. 
    
    :param obs: Sequence of observations
    :type obs: Array
    :param initial_params: Initial guesses for the parameters
    :type initial_params: HiddenMarkovModel
    :param max_iter: Maximum number of iterations
    :type max_iter: int
    :param epsilon: Convergence threshold
    :type epsilon: float
    :param mode: Flag to indicate if log probabilities should be used. Can be either `log` or `regular`
    :type mode: str
    :return: Returns the parameter estimate as well as the likelihood and residual sequences. (log or regular probabilities based on `mode`)
    :rtype: IterationState
    '''
    
    _require_x64()
    
    if mode == 'log':
        if not initial_params.is_log:
            initial_params = initial_params.to_log()
        
    elif mode == 'regular':
        if initial_params.is_log:
            initial_params = initial_params.to_prob()
    else:
        raise ValueError('mode argument must be either "log" or "regular"!')
    
    # Peform checks if sizes match up and ensure obs always has a leading axis
    obs, mu = standardize_shapes(obs, initial_params)

    # If mulitple observation sequences, but only a single mu distribution are passed, 
    # use a shared mu distribution for all sequences
    shared_mu = (len(obs) > 1) and (initial_params.mu.ndim == 1)

    initial_params = HiddenMarkovParameters(initial_params.T, initial_params.O, mu, initial_params.is_log)

    final_state = _baum_welch_impl(obs, initial_params, max_iter, epsilon, shared_mu, mode)

    if shared_mu:
        final_state = final_state.replace_mu(jnp.mean(final_state.params.mu, axis=0))

    return final_state.squeeze()

@wrapped_jit(static_argnames=["max_iter", "epsilon", "mode", "shared_mu"])
def _baum_welch_impl(obs: Array,
        initial_params: HiddenMarkovParameters,
        max_iter: int = 100,
        epsilon: float = 1e-6,
        shared_mu: bool = True,
        mode: str = 'log') -> IterationState:
    '''This implementation already expects the initial state distributions mu and obs to have a leading axis of
    the same lenght.'''

    m_obs = initial_params.O.shape[-1]
    n_mu = initial_params.mu.shape[0]
    is_log = mode == 'log'

    update_parameters = _maximization_step_log if is_log else _maximization_step

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
            gamma, xi = forward_backward(obs, inner_carry.params, mode=mode, squeeze=False)
            
            _log_llhood = log_likelihood(obs, inner_carry.params)
            log_llhood = jnp.sum(_log_llhood)


            # Maximization - step
            # Initial state probabilities
            if shared_mu:
                # Use a shared average mu estimate for all sequences
                mu = jnp.mean(gamma[:, 0], axis=0)[None, ...].repeat(n_mu, axis=0)
            else:
                # Separate mu estimates for each sequence
                mu = gamma[:, 0]

            gamma = jnp.concat(gamma, axis=0)
            xi = jnp.concat(xi, axis=0)

            # Maximization - step
            # Transition and observation probabilities
            T, O = update_parameters(obs, gamma, xi, m_obs)
            updated = HiddenMarkovParameters(T, O, mu, is_log=is_log)

            
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