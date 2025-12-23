from typing import NamedTuple
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array, vmap

from ..util import wrapped_jit, normalize_rows
from ..models import HiddenMarkovModel
from .forward_backward import forward_backward

class IterationState(NamedTuple):
    params: HiddenMarkovModel
    residual: float
    iteration: int
    terminated: bool
    

@wrapped_jit(static_argnames=["max_iter", "epsilon", "mode"])
def baum_welch(obs: Array,
        initial_params: HiddenMarkovModel,
        max_iter=100,
        epsilon=1e-4,
        mode='log') -> IterationState:
    
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

    parallel_mode = len(obs.shape) > 1

    m = initial_params.O.shape[-1]

    def iteration(carry: tuple[HiddenMarkovModel, ], _: None) -> tuple[Array, Array]:

        T, O = carry

        # Expectation - step
        # (Forward-Backward algorithm for estimating probabilities 
        # of the latent variables (states) given the observations)
        if parallel_mode:
            gamma, xi = vmap(
                lambda _o: forward_backward(_o, T, O, mu, mode='regular'))(obs)

            gamma = jnp.concat(gamma, axis=0)
            xi = jnp.concat(xi, axis=0)
        else:
            gamma, xi = forward_backward(obs, T, O, mu, mode='regular')

        # Maximization - step
        # Average over all time steps and normalize along rows => new estimate for T
        T = jnp.sum(xi, axis=0)
        T = normalize_rows(T)  # T / jnp.sum(T, axis=-1)[..., None]

        O = lax.map(lambda o: jnp.sum(
            (obs.ravel() == o)[:, None] * gamma, axis=0), jnp.arange(m)).T
        O = O / jnp.sum(gamma, axis=0)[..., None]

        return (T, O), (T, O)

    _, (T_seq, O_seq) = lax.scan(
        iteration,
        init=(T_0.copy(), O_0.copy()),
        length=max_iter
    )

    # Pick the first index where the max difference between subsequent iterations is below epsilon
    convergence_idx = jnp.argmax(
        jnp.max(jnp.abs(jnp.diff(T_seq, 1, axis=0)), axis=(1, 2)) < epsilon)

    return T_seq[convergence_idx], O_seq[convergence_idx]



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

                gamma = jnp.concat(gamma, axis=0)
                xi = jnp.concat(xi, axis=0)
            else:
                gamma, xi = forward_backward(obs, inner_carry.params, mode='log')

            # Maximizaton - step
            # Average over all time steps and normalize along rows => new estimate for T
            T = logsumexp(xi, axis=0)
            T -= logsumexp(T, axis=-1)[..., None]

            O = lax.map(lambda o: logsumexp(
                jnp.log(obs.ravel() == o)[:, None] + gamma, axis=0), jnp.arange(m)).T
            O -= logsumexp(gamma, axis=0)[..., None]

            mu = gamma[0]

            residual_T = jnp.max(jnp.abs(jnp.exp(T) - jnp.exp(inner_carry.params.T)))
            residual_O = jnp.max(jnp.abs(jnp.exp(O) - jnp.exp(inner_carry.params.O)))
            residual_mu = jnp.max(jnp.abs(jnp.exp(mu) - jnp.exp(inner_carry.params.mu)))

            residual = jnp.max(jnp.array([residual_T, residual_O, residual_mu]))

            updated_hmm = HiddenMarkovModel(T, O, mu, is_log=True)

            return lax.cond(
                jnp.any(jnp.isnan(T)) | jnp.any(jnp.isnan(O)),
                lambda: IterationState(updated_hmm, residual, inner_carry.iteration, True),#(*carry[:-1], True),
                lambda: IterationState(updated_hmm, residual, inner_carry.iteration + 1, False)

            )

        hmm, residual, n, terminated = carry

        carry = lax.cond(
            terminated | (residual < epsilon),
            lambda x: x,
            perform_step,
            IterationState(hmm, residual, n, terminated)

        )

        return carry, None

    final_state, _ = lax.scan(
        iteration,
        init=IterationState(initial_params, epsilon + 1, 0, False),
        length=max_iter
    )
    
    return final_state
    # T, O, residual, n, terminated = result

    # # Pick the first index where the max difference between subsequent iterations is below epsilon
    # convergence_idx = jnp.argmax(
    #     jnp.max(jnp.abs(jnp.diff(jnp.exp(T_seq), 1, axis=0)), axis=(1, 2)) < epsilon)
    # return (.exp(T), jnp.exp(O), residual, n, terminated) #jnp.exp(T_seq[convergence_idx]), jnp.exp(O_seq[convergence_idx])

    # def iteration(carry: tuple[Array, Array], _: None) -> tuple[Array, Array]:

    #     T, O = carry

    #     # E - step
    #     # (Forward-Backward algorithm for estimating transition and emission probabilities)
    #     if parallel_mode:
    #         gamma, xi = vmap(
    #             lambda _o: forward_backward_log(_o, T, O, mu))(obs)

    #         gamma = jnp.concat(gamma, axis=0)
    #         xi = jnp.concat(xi, axis=0)
    #     else:
    #         gamma, xi = forward_backward_log(obs, T, O, mu)

    #     # M - step
    #     # Average over all time steps and normalize along rows => new estimate for T
    #     T = logsumexp(xi, axis=0)
    #     T -= logsumexp(T, axis=-1)[..., None]

    #     O = lax.map(lambda o: logsumexp(
    #         jnp.log(obs.ravel() == o)[:, None] + gamma, axis=0), jnp.arange(m)).T
    #     O -= logsumexp(gamma, axis=0)[..., None]

    #     return (T, O), (T, O)

    # _, (T_seq, O_seq) = lax.scan(
    #     iteration,
    #     init=(T_0.copy(), O_0.copy()),
    #     length=max_iter
    # )

    # # Pick the first index where the max difference between subsequent iterations is below epsilon
    # convergence_idx = jnp.argmax(
    #     jnp.max(jnp.abs(jnp.diff(jnp.exp(T_seq), 1, axis=0)), axis=(1, 2)) < epsilon)

    # return jnp.exp(T_seq[convergence_idx]), jnp.exp(O_seq[convergence_idx])