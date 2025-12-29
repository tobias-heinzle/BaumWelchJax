from jax import Array
import jax.numpy as jnp

def normalize_rows(arr: Array) -> Array:
    '''
    Normalizes array along its last axis, while coercing `nan` values to 0 
    
    :param arr: Array to normalize
    :type arr: Array
    :return: Normalized array such that entries sum to one along the last axis.
    :rtype: Array
    '''
    sum_vec = jnp.sum(arr, axis=-1)
    return jnp.nan_to_num(arr / sum_vec[..., None], 0.0)

