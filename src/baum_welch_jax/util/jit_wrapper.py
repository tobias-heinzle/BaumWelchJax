from functools import wraps

import jax

def wrapped_jit(*jit_args, **jit_kwargs):
    """Like jax.jit but preserves signature, docstring, and type hints."""
    def decorator(func):
        # Apply jax.jit but keep original metadata
        jitted = jax.jit(func, *jit_args, **jit_kwargs)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return jitted(*args, **kwargs)
        return wrapper
    return decorator