from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax import Array


def normalize_rows(vec: Array) -> Array:
    sum_vec = jnp.sum(vec, axis=-1)
    return jnp.nan_to_num(vec / sum_vec[..., None], 0.0)


@jax.jit
def forward_backward(obs: Array, T: Array, O: Array, mu: Array) -> tuple[Array, Array]:
    n = mu.shape[0]
    t_max = len(obs)

    # Initialize forward probabilities
    alpha_0 = mu * O[:, obs[0]]
    alpha_0 = normalize_rows(alpha_0)  # alpha_0 / jnp.sum(alpha_0)

    # Initialize backward probabilities
    beta_t_max = jnp.ones(n) / n

    def step(carry, t):
        alpha, beta = carry

        alpha = (alpha @ T) * O[:, obs[t]]
        beta = T @ (O[:, obs[t_max - t]] * beta)

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

    # If alpha is normalized in each step of the algorithm anyways, we can avoid this
    # likelihood = jnp.sum(alpha[t_max - 1])
    # jax.debug.print("{}", likelihood)
    gamma = (alpha * beta)  # / likelihood

    # Calculation of the xi tensor involves taking the outer product of alpha and O * beta
    # for each combination of alpha_t and beta_t+1
    obs_probs = jnp.take(O, obs[1:], axis=1).T
    xi = jnp.einsum("ij, ik->ijk", alpha[:-1], beta[1:] * obs_probs)

    # and then multiplying each slice componentwise with _T
    xi = xi * T[None, ...]

    return gamma, xi


@partial(jax.jit, static_argnames=["max_iter", "epsilon"])
def baum_welch(
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
            gamma, xi = jax.vmap(
                lambda _o: forward_backward(_o, T, O, mu))(obs)

            gamma = jnp.concat(gamma, axis=0)
            xi = jnp.concat(xi, axis=0)
        else:
            gamma, xi = forward_backward(obs, T, O, mu)

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
