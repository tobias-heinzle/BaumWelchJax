import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax import Array, vmap

from ..util import wrapped_jit, normalize_rows
from .forward_backward import forward_backward

@wrapped_jit(static_argnames=["max_iter", "epsilon", "mode"])
def baum_welch(obs: Array,
        T_0: Array,
        O_0: Array,
        mu: Array,
        max_iter=100,
        epsilon=1e-4,
        mode='log') -> tuple[Array, Array]:
    
    if mode == 'log':
        return _baum_welch_log(obs, T_0, O_0, mu, max_iter, epsilon)
    elif mode == 'regular':
        return _baum_welch(obs, T_0, O_0, mu, max_iter, epsilon)
    else:
        raise ValueError('mode argument must be either "log" or "regular"!')
    

@wrapped_jit(static_argnames=["max_iter", "epsilon"])
def _baum_welch(
        obs: Array,
        T_0: Array,
        O_0: Array,
        mu: Array,
        max_iter=100,
        epsilon=1e-4) -> tuple[Array, Array]:

    parallel_mode = len(obs.shape) > 1

    m = O_0.shape[-1]

    def iteration(carry: tuple[Array, Array], _: None) -> tuple[Array, Array]:

        T, O = carry

        # E - step
        # (Forward-Backward algorithm for estimating transition and emission probabilities)
        if parallel_mode:
            gamma, xi = vmap(
                lambda _o: forward_backward(_o, T, O, mu, mode='regular'))(obs)

            gamma = jnp.concat(gamma, axis=0)
            xi = jnp.concat(xi, axis=0)
        else:
            gamma, xi = forward_backward(obs, T, O, mu, mode='regular')

        # M - step
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
        T_0: Array,
        O_0: Array,
        mu: Array,
        max_iter=100,
        epsilon=1e-4) -> tuple[Array, Array, Array, int, bool]:

    parallel_mode = len(obs.shape) > 1

    m = O_0.shape[-1]

    def iteration(
            carry: tuple[Array, Array, Array, int, bool],
            _: None
            ) -> tuple[tuple[Array, Array, Array, int, bool], None]:


        def perform_step(
                carry: tuple[Array, Array, Array, int, bool]
                ) -> tuple[Array, Array, Array, int, bool]:
            
            T, O, _, n, _ = carry

            # E - step
            # (Forward-Backward algorithm for estimating transition and emission probabilities)
            if parallel_mode:
                gamma, xi = vmap(
                    lambda _o: forward_backward(_o, T, O, mu, mode='log'))(obs)

                gamma = jnp.concat(gamma, axis=0)
                xi = jnp.concat(xi, axis=0)
            else:
                gamma, xi = forward_backward(obs, T, O, mu, mode='log')

            # M - step
            # Average over all time steps and normalize along rows => new estimate for T
            T = logsumexp(xi, axis=0)
            T -= logsumexp(T, axis=-1)[..., None]

            O = lax.map(lambda o: logsumexp(
                jnp.log(obs.ravel() == o)[:, None] + gamma, axis=0), jnp.arange(m)).T
            O -= logsumexp(gamma, axis=0)[..., None]

            residual_T = jnp.max(jnp.abs(jnp.exp(T) - jnp.exp(carry[0])))
            residual_O = jnp.max(jnp.abs(jnp.exp(O) - jnp.exp(carry[1])))

            residual = jnp.max(jnp.array([residual_T, residual_O]))

            return lax.cond(
                jnp.any(jnp.isnan(T)) | jnp.any(jnp.isnan(O)),
                lambda: (T, O, residual, n + 1, True),#(*carry[:-1], True),
                lambda: (T, O, residual, n + 1, False)

            )

        T, O, residual, n, terminated = carry

        carry = lax.cond(
            terminated | (residual < epsilon),
            lambda x: x,
            perform_step,
            (T, O, residual, n, terminated)

        )

        return carry, None

    result, _ = lax.scan(
        iteration,
        init=(T_0.copy(), O_0.copy(), epsilon + 1, 0, False),
        length=max_iter
    )
    
    T, O, residual, n, terminated = result

    # # Pick the first index where the max difference between subsequent iterations is below epsilon
    # convergence_idx = jnp.argmax(
    #     jnp.max(jnp.abs(jnp.diff(jnp.exp(T_seq), 1, axis=0)), axis=(1, 2)) < epsilon)
    return (jnp.exp(T), jnp.exp(O), residual, n, terminated) #jnp.exp(T_seq[convergence_idx]), jnp.exp(O_seq[convergence_idx])

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