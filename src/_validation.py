"""Option checks shared by ``make_bihalofit`` and ``make_projected_bihalofit``: ``pk_source``,
``term``, ``k_lin``, and the report of a failed k_NL / n_eff solve.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ._pk_backends import _VALID_SOURCES

_VALID_TERMS = ("full", "1h", "3h", "tree")


def check_pk_options(pk_source, term, k_lin):
    """Raise ``ValueError`` for an unknown ``pk_source`` or ``term``, or a ``k_lin`` the backend cannot take.

    Parameters
    ----------
    pk_source : str
        Linear P(k) backend name.
    term : str
        Bispectrum term, one of ``'full'``, ``'1h'``, ``'3h'``, ``'tree'``.
    k_lin : array-like or None
        User linear-P(k) grid; must be None for ``'baccoemu'``.

    Returns
    -------
    None
    """
    if pk_source not in _VALID_SOURCES:
        raise ValueError(
            f"pk_source must be one of {_VALID_SOURCES} (got {pk_source!r})."
        )
    if term not in _VALID_TERMS:
        raise ValueError(f"term must be one of {_VALID_TERMS} (got {term!r}).")
    if k_lin is not None and pk_source == "baccoemu":
        raise ValueError(
            "k_lin is not supported with pk_source='baccoemu' (its output lives on a grid fixed by the "
            "trained emulator) -- use 'eh', 'eh_99', or 'syren'."
        )


def check_solver(B_out, resid_out, *, tol, validate, N_R, jdtype, over):
    """Check the k_NL/n_eff solve behind ``B_out`` and return ``B_out``, with NaN where the residual exceeds ``tol``.

    ``B_out`` has the triangles on its last axis and ``resid_out`` is the Newton residual of each of the leading entries.
    A bracket failure (sigma^2(R) never reaches 1, so ``B_out`` is already NaN) raises if ``validate``; a residual above
    ``tol`` raises if ``validate`` and is set to NaN otherwise. ``over`` names what the leading axes index, for the
    message. Under a JAX trace nothing concrete can be checked and ``B_out`` is returned as it is.

    Parameters
    ----------
    B_out : (..., N_tri)
        Bispectrum, triangles on the last axis.
    resid_out : shape of ``B_out[..., 0]``
        Newton residual of the k_NL/n_eff solve per leading entry.
    tol : float or None
        Residual threshold; None skips the residual check.
    validate : bool
        Raise on a failure (True) or set NaN (False).
    N_R : int
        Newton iteration count, quoted in the error message.
    jdtype : JAX dtype
        dtype of the returned array when entries are set to NaN.
    over : str
        Name of the leading axes, quoted in the error message.

    Returns
    -------
    B : shape of ``B_out``
        ``B_out``, with NaN where the residual exceeds ``tol`` (when not raising).
    """
    try:
        B_np = np.asarray(B_out)
        resid_np = np.asarray(resid_out) if tol is not None else None
    except jax.errors.TracerArrayConversionError:
        return B_out

    bad_bracket = np.isnan(B_np).any(axis=-1)
    if bad_bracket.any() and validate:
        raise ValueError(
            f"sigma^2(R, z) never reaches 1.0 for {int(bad_bracket.sum())} {over} combination(s) -- "
            "sigma8_cold is too small or [R_min, R_max] does not bracket k_NL"
        )
    if tol is None:
        return B_out  # only the bracket check of `validate` was requested

    bad = (
        resid_np > tol
    ) & ~bad_bracket  # bracket failures are NaN already, not a tol problem
    if not bad.any():
        return B_out
    if validate:
        raise ValueError(
            f"k_NL/n_eff Newton residual reached {resid_np[bad].max():.3g}, above the requested "
            f"tol={tol:g}, for {int(bad.sum())} {over} combination(s) at N_R={N_R}. Pass a larger N_R "
            "directly -- tol only sizes the iteration count up front, it does not re-solve."
        )
    return jnp.asarray(np.where(bad[..., None], np.nan, B_np), dtype=jdtype)
