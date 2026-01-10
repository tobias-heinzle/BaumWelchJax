from typing import Callable

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array

from ..util import wrapped_jit, normalize_rows, standardize_shapes
from ..models import HiddenMarkovParameters, IterationState, FreezeConfig, FreezeMasks, assert_valid_hmm
from .forward_backward import forward_backward
from .likelihoods import log_likelihood
from .._precision import _warn_if_fp32


# TODO: How to handle ragged seqeunces of different length?
#       In principle, also final state distributions could be provided and estimated during the forward backward pass.
#       Look into this in more detail!

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
        # This log will be -inf at a given index if obs != o there!
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
    

def baum_welch(obs: Array,
        initial_params: HiddenMarkovParameters,
        max_iter: int = 100,
        tol: float = 1e-6,
        check_ascent: bool = False,
        ascent_tol: float = 0.0,
        mode: str = 'log',
        freeze_config: FreezeConfig | FreezeMasks = FreezeConfig()
        ) -> IterationState:
    '''
    Implementation of expectation maximization for hidden Markov models.
    `baum_welch` can only be used with x64 precision. 
    
    :param obs: Sequence of observations
    :type obs: Array
    :param initial_params: Initial guesses for the parameters
    :type initial_params: HiddenMarkovModel
    :param max_iter: Maximum number of iterations
    :type max_iter: int
    :param tol: Tolerance thershold for convergence
    :type tol: float
    :param check_ascent: If `True` iteration is stopped once objective decreases for more then `ascent_tol`.
    :type check_ascent: bool
    :param ascent_tol: Tolerance threshold for the objective increases when ascent checking is enabled
    :type ascent_tol: float
    :param mode: Flag to indicate if log probabilities should be used. Can be either `log` or `regular`
    :type mode: str
    :return: Returns the parameter estimate as well as the computed likelihoods and residual sequences. (log or regular probabilities based on `mode`)
    :rtype: IterationState
    '''
    
    _warn_if_fp32()

    try:
        assert_valid_hmm(initial_params)
    except ValueError as exc:
        raise ValueError('`initial_params` is not a valid set of parameters!') from exc

    
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

    @jax.jit
    def stop_criterion(state: IterationState) -> bool:
        n_step = state.iterations
        log_llhoods = state.log_likelihoods

        # The log likelihood difference of two subsequent iterations is used to determine convergence
        is_converged = (jnp.abs(log_llhoods[n_step - 1] - log_llhoods[n_step - 2])) < tol

        if check_ascent:
            has_decreased = log_llhoods[n_step - 1] < log_llhoods[n_step - 2] - ascent_tol
            return jnp.any(jnp.array([state.terminated, has_decreased, is_converged]))
        
        return jnp.any(jnp.array([state.terminated, is_converged]))

    # Set up freeze configuration
    # This freeze mask setup is not finished yet!
    # Works for fixing an entire matrix or row, but does not perform correctly for
    # fixing of single indices!
    def is_correct(masks: FreezeMasks) -> bool:
        # Check if the freeze mask has the supported structure
        checked_mask = jax.tree.map(lambda leaf: 
            jnp.all(
                jnp.all(leaf == True, axis=-1) 
                | jnp.all(leaf == False, axis=-1)
            ),
            masks)
        
        return checked_mask.T and checked_mask.O and checked_mask.mu
    
    # TODO: Adapt the algorithm must to correctly handle single frozen paramters. 
    # In particular, the re-normalization of the new parameters must be performed considering only
    # the remaining probability mass that is not yet used by the frozen parameters.
    if isinstance(freeze_config, FreezeConfig):
        freeze_masks = freeze_config.create_masks(initial_params)
    else:
        if not is_correct(freeze_config):
            raise ValueError('Invalid `freeze_config`: Only entire rows can be frozen, not single entries!')

        freeze_masks = freeze_config

    

    final_state = _baum_welch_impl(
        obs, 
        initial_params, 
        stop_criterion, 
        max_iter, 
        shared_mu, 
        mode,
        freeze_masks
        )

    if shared_mu:
        final_state = final_state.replace_mu(jnp.mean(final_state.params.mu, axis=0))

    final_state = final_state.squeeze()
    assert_valid_hmm(final_state.params)

    return final_state

@wrapped_jit(static_argnames=["stop_criterion", "max_iter", "mode", "shared_mu"])
def _baum_welch_impl(obs: Array,
        initial_params: HiddenMarkovParameters,
        stop_criterion: Callable[[IterationState], bool],
        max_iter: int,
        shared_mu: bool,
        mode: str,
        freeze_masks: FreezeMasks) -> IterationState:
    '''This implementation already expects the initial state distributions mu and obs to have a leading axis of
    the same lenght.'''

    m_obs = initial_params.O.shape[-1]
    n_obs = obs.shape[0]
    n_mu = initial_params.mu.shape[0]
    is_log = (mode == 'log')

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


            # Maximization - step
            # Initial state probabilities
            if shared_mu:
                # Use a shared average mu estimate for all sequences 

                # TODO: How should this estimate be weighted? 
                # Is it really correct to just take the mean here?
                
                if is_log:
                    mu = logsumexp(gamma[:, 0], axis=0, keepdims=True).repeat(n_mu, axis=0) - jnp.log(n_obs)
                else:
                    mu = jnp.mean(gamma[:, 0], axis=0, keepdims=True).repeat(n_mu, axis=0)

            else:
                # Separate mu estimates for each sequence
                mu = gamma[:, 0]

            gamma = jnp.concat(gamma, axis=0)
            xi = jnp.concat(xi, axis=0)

            # Maximization - step
            # Transition and observation probabilities
            T, O = update_parameters(obs, gamma, xi, m_obs)
            updated = HiddenMarkovParameters(
                jnp.where(freeze_masks.T, inner_carry.params.T, T), 
                jnp.where(freeze_masks.O, inner_carry.params.O, O), 
                jnp.where(freeze_masks.mu, inner_carry.params.mu, mu), 
                is_log=is_log)

            
            residual = _compute_residual(updated, inner_carry.params, mode=mode)    
            residuals = inner_carry.residuals.at[inner_carry.iterations].set(residual)

            _log_llhood = log_likelihood(obs, updated.to_prob() if is_log else updated)
            log_llhood = jnp.sum(_log_llhood)
            log_llhoods = inner_carry.log_likelihoods.at[inner_carry.iterations].set(log_llhood)

            return lax.cond(
                # Stop the iteration upon detection of NaN values!
                jnp.any(jnp.isnan(updated.T)) | jnp.any(jnp.isnan(updated.O) | jnp.any(jnp.isnan(updated.mu))),
                lambda: IterationState(inner_carry.params, log_llhoods, residuals, inner_carry.iterations, True),
                lambda: IterationState(updated, log_llhoods, residuals, inner_carry.iterations + 1, False)

            )

        carry = lax.cond(
            stop_criterion(carry),
            lambda x: x,
            perform_step,
            carry
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