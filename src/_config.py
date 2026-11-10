"""Precision policy.

The builders default to float32: values agree with float64 to < 0.02% over the documented range,
and it is ~7x faster on the GPU. float32 gradients are checked only at neutrino_mass = 0; use float64
for other gradient work. The package never sets jax_enable_x64: call enable_x64() or set
JAX_ENABLE_X64=1, and requesting float64 without it raises.
"""

import numpy as np
import jax
import jax.numpy as jnp


def enable_x64():
    """Enable JAX's float64 mode process-wide.

    A convenience for scripts and notebooks::

        import bihalofit_jax
        bihalofit_jax.enable_x64()

    Must be called before any JAX computation. Library code should never call
    this; an application should, or should set the equivalent
    ``JAX_ENABLE_X64=1`` environment variable / ``jax.config`` flag itself.

    Takes no arguments and returns None.
    """
    jax.config.update("jax_enable_x64", True)


def resolve_dtype(dtype):
    """Map a user-supplied ``dtype=`` to the JAX dtype the kernel will use.

    Parameters
    ----------
    dtype : numpy dtype or dtype-like
        ``np.float64`` selects float64; anything else selects float32.

    Returns
    -------
    jdtype : ``jnp.float32`` or ``jnp.float64``

    Raises if
    float64 was requested while JAX's x64 mode is off: without this check
    JAX would silently demote every float64 array to float32 and return a
    quietly inaccurate result rather than an error. float32 primal values
    are validated to <0.02% of float64's; float32 *gradients* need their
    own validation per combination (see this module's docstring).
    """
    if np.dtype(dtype) == np.float64:
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "bihalofit-jax needs JAX's float64 mode for "
                "dtype=np.float64, but jax_enable_x64 is off, so float64 "
                "arrays would be silently demoted to float32.\n\n"
                "Enable it once, before any JAX computation:\n"
                "    import bihalofit_jax\n"
                "    bihalofit_jax.enable_x64()\n"
                "or equivalently jax.config.update('jax_enable_x64', True), "
                "or set JAX_ENABLE_X64=1 in the environment.\n\n"
                "Or drop the dtype= argument: float32 is the default, "
                "needs no flag, and is validated to agree with float64 to "
                "<0.02% on primal values across the documented prior box "
                "(see this module's docstring); gradients need their own "
                "case-by-case validation."
            )
        return jnp.float64
    return jnp.float32
