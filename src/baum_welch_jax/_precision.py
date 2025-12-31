import warnings
import jax

class PrecisionWarning(UserWarning):
    """Emitted to inform user about JAX running in single precision."""
    pass

def _warn_if_fp32(*, stacklevel=2):
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "Running in single precision (float32). "
            "Double precision is recommended.\n"
            "It must be specified when importing JAX:\n\n"
            "  import jax\n"
            "  jax.config.update('jax_enable_x64', True)\n",
            PrecisionWarning,
            stacklevel=stacklevel,
        )