from jax import Array
import jax.numpy as jnp

def normalize_rows(vec: Array) -> Array:
    sum_vec = jnp.sum(vec, axis=-1)
    return jnp.nan_to_num(vec / sum_vec[..., None], 0.0)

