import jax.numpy as jnp
import jax


def logsumexp(x):
    c = jnp.max(x)

    # jax.debug.print("x: {}", x, ordered=True)
    # jax.debug.print("c: {}", c, ordered=True)
    # jax.debug.print("x -c: {}", x - c, ordered=True)
    # jax.debug.print("exp(x-c): {}", jnp.exp(x - c), ordered=True)
    # jax.debug.print("sum(exp(x-c)): {}", jnp.sum(jnp.exp(x - c)), ordered=True)
    # jax.debug.print("log(sum(exp(x-c))): {}",
    #                 jnp.log(jnp.sum(jnp.exp(x - c))), ordered=True)

    diff = jnp.nan_to_num(x - c, -jnp.inf)
    return c + jnp.log(jnp.sum(jnp.exp(diff)))
