from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax import Array


def normalize_rows(vec: Array) -> Array:
    sum_vec = jnp.sum(vec, axis=-1)
    return lax.cond(
        jnp.allclose(sum_vec, 0.0),
        lambda: jnp.zeros_like(vec),
        lambda: vec / sum_vec[..., None]
    )


@partial(jax.jit, static_argnames=["max_iter", "epsilon"])
def forward_backward(
        obs: Array,
        mu: Array,
        T_0: Array,
        O_0: Array,
        max_iter=100,
        epsilon=1e-4) -> tuple[Array, Array]:

    t_max = len(obs)
    n, m = O_0.shape

    def iteration(carry: tuple[Array, Array], _: None) -> tuple[Array, Array]:

        _T, _O = carry

        # Initial forward probabilities for the loop
        alpha_0 = mu * _O[:, obs[0]]
        alpha_0 = normalize_rows(alpha_0)

        # initialize backward probabilities
        beta_t_max = jnp.ones(n) / n

        t_range = jnp.arange(1, t_max)

        def step(carry, t):
            alpha, beta = carry

            alpha = (alpha @ _T) * _O[:, obs[t]]
            beta = _T @ (_O[:, obs[t_max - t]] * beta)

            alpha = normalize_rows(alpha)
            beta = normalize_rows(beta)

            return (alpha, beta), (alpha, beta)

        _, (alpha, beta) = lax.scan(
            step,
            (alpha_0, beta_t_max),
            t_range
        )

        alpha = jnp.concat([alpha_0[None, :], alpha])
        beta = jnp.concat([beta_t_max[None, :], beta])

        beta = jnp.flip(beta, axis=0)

        likelihood = jnp.sum(alpha[t_max - 1])
        gamma = (alpha * beta) / (likelihood + epsilon)

        # Calculation of the xi tensor involves taking the outer product of alpha and O * beta
        # for each combination of alpha_t and beta_t+1
        obs_probs = jnp.take(_O, obs[1:], axis=1).T
        xi = jnp.einsum("ij, ik->ijk", alpha[:-1], beta[1:] * obs_probs)

        # and then multiplying each slice componentwise with _T
        xi = xi * _T[None, ...]

        # Now by averaging over all time steps and normalizing along the rows,
        # a new estimate for T is obtained
        _T = jnp.sum(xi, axis=0)
        _T = normalize_rows(_T)

        _O = lax.map(lambda o: jnp.sum(
            (obs == o)[:, None] * gamma, axis=0), jnp.arange(m)).T
        _O = normalize_rows(_O)
        # _O = _O / np.sum(gamma, axis=0)[..., None]

        return (_T, _O), (_T, _O)

    (_T, _O), (seq_T, seq_O) = lax.scan(
        iteration,
        init=(T_0.copy(), O_0.copy()),
        length=max_iter
    )

    # Pick the first index where the max difference between subsequent iterations is below epsilon
    convergence_idx = jnp.argmax(
        jnp.max(jnp.abs(jnp.diff(seq_T, 1, axis=0)), axis=(1, 2)) < epsilon)

    return seq_T[convergence_idx], seq_O[convergence_idx]
