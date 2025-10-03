import jax.numpy as jnp
import jax


def logsumexp(x):
    """
    Performs the logsumexp trick for normalizing logprobabilities.
    See https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    c = jnp.max(x)
    diff = jnp.nan_to_num(x - c, -jnp.inf)
    return c + jnp.log(jnp.sum(jnp.exp(diff)))
