"""k_NL (sigma^2(R_NL) = 1) and n_eff from P_L(k), by a safeguarded Newton solve."""

import numpy as np
import jax.numpy as jnp

from ._pk_backends import _STANDARD_K_GRID, _build_sigma8_weights, _trapezoid_dk

_solver_cache: dict = {}  # keyed on N_k (or k_lin identity for a custom grid)


def _get_solver(k_lin=None, jdtype=jnp.float64):
    """Constants for the k_NL/n_eff solve, ready to hand to the kernel.

    Parameters
    ----------
    k_lin : (N_k,) array-like, optional
        Linear P(k) grid in h/Mpc (default ``_STANDARD_K_GRID``).
    jdtype : JAX dtype, optional

    Returns
    -------
    k2, k4 : (N_k,) jax array
        k^2 and k^4 on the grid.
    dk_over_2pi2 : (N_k,) jax array
        Trapezoid weights divided by 2 pi^2.
    w8 : (N_k,) jax array
        Top-hat (R = 8 Mpc/h) quadrature weights, so that sigma8(z)^2 is proportional to
        sum(w8 * P_L(k, z)).

    Only the numpy precompute is cached, and on the grid alone: it is
    device- and dtype-agnostic that way. The conversion below is not worth
    caching, since callers run it inside a jit trace where it constant-folds
    into the compiled executable.
    """
    if k_lin is None:
        k_lin = _STANDARD_K_GRID
    k_lin = np.asarray(k_lin)
    N_k = len(k_lin)
    solver_key = N_k if k_lin is _STANDARD_K_GRID else k_lin.tobytes()
    if solver_key not in _solver_cache:
        k = k_lin.astype(np.float64)
        dk = _trapezoid_dk(k)
        _solver_cache[solver_key] = (
            k**2,
            k**4,
            dk / (2.0 * np.pi**2),
            _build_sigma8_weights(k),
        )
    k2, k4, dk_over_2pi2, w8 = _solver_cache[solver_key]
    return (
        jnp.asarray(k2, dtype=jdtype),
        jnp.asarray(k4, dtype=jdtype),
        jnp.asarray(dk_over_2pi2, dtype=jdtype),
        jnp.asarray(w8, dtype=jdtype),
    )


def _sigma2_k4int(R, k2, k4, dk_over_2pi2, PL):
    """Gaussian-filtered sigma2(R) and k4int(R), the k^4-weighted integral, at R.

    Parameters
    ----------
    R : (N_cosmo,)
        Smoothing radius in Mpc/h.
    k2, k4, dk_over_2pi2 : (N_k,)
        From _get_solver.
    PL : (N_cosmo, N_k)
        Linear P(k) in (Mpc/h)^3.

    Returns
    -------
    sigma2, k4int : (N_cosmo,)
    """
    base = dk_over_2pi2[None, :] * jnp.exp(-jnp.outer(R**2, k2))
    sigma2 = jnp.sum(base * k2[None, :] * PL, axis=1)
    k4int = jnp.sum(base * k4[None, :] * PL, axis=1)
    return sigma2, k4int


def _tol_to_niter(tol):
    """tol -> fixed Newton iteration count (static int, computed pre-trace).

    Newton converges quadratically past a ~4-iteration warm-up; +4 margin
    covers the quadratic-model's own error and pushes past "just converged"
    to bit-identical reproducibility against a manually-chosen N_R.

    Parameters
    ----------
    tol : float
        Target relative error on k_NL and n_eff.

    Returns
    -------
    n_iter : int
        At most 30.
    """
    N_WARMUP, E_WARMUP = 4, 1e-2
    MAX_N_ITER = 30  # tol below ~float64 eps is unreachable regardless
    if tol >= E_WARMUP:
        n_extra = 0
    else:
        n_extra = int(np.ceil(np.log2(np.log(tol) / np.log(E_WARMUP))))
    return min(N_WARMUP + n_extra + 4, MAX_N_ITER)


def _resolve_N_R(N_R, tol, default=15):
    """Resolve Newton iteration count from N_R/tol (mutually exclusive,
    detected by N_R != default). N_R<2 is rejected -- too few iterations
    risk landing on an n_eff that overflows the fitting formula.

    Parameters
    ----------
    N_R : int
        User iteration count.
    tol : float or None
        User tolerance; if set, N_R must equal ``default``.
    default : int, optional
        The builders' default N_R.

    Returns
    -------
    n_iter : int
    """
    if tol is not None and N_R != default:
        raise ValueError("tol and N_R are mutually exclusive; only one may be set.")
    if tol is not None:
        return _tol_to_niter(tol)
    if N_R < 2:
        raise ValueError(
            f"N_R={N_R} is too small to be numerically safe: with the wide "
            "default R_min=1e-6, a single Newton step can land on a wildly "
            "wrong n_eff. Use N_R >= 2, or narrow R_min."
        )
    return N_R


def _solve_knl_neff(PL, k2, k4, dk_over_2pi2, R_min, R_max, n_iter):
    """Locate k_NL and n_eff via safeguarded (bisection-backed) Newton on
    _sigma2_k4int, unrolled for n_iter static steps.

    Parameters
    ----------
    PL : (N_cosmo, N_k)
        Linear P(k) in (Mpc/h)^3.
    k2, k4, dk_over_2pi2 : (N_k,)
        From _get_solver.
    R_min, R_max : float
        Bracket in Mpc/h.
    n_iter : int
        Static number of Newton steps.

    Returns
    -------
    knl : (N_cosmo,)
        1/R_NL in h/Mpc.
    neff : (N_cosmo,)
        Effective spectral index.
    crossed : (N_cosmo,) bool
        False (knl/neff meaningless) wherever sigma2 never reaches 1 on [R_min, R_max].
    resid : (N_cosmo,)
        Relative change in R on the last step.
    """
    N_cosmo = PL.shape[0]
    R_lo = jnp.full((N_cosmo,), R_min, dtype=PL.dtype)
    R_hi = jnp.full((N_cosmo,), R_max, dtype=PL.dtype)
    s_lo, _ = _sigma2_k4int(R_lo, k2, k4, dk_over_2pi2, PL)
    s_hi, _ = _sigma2_k4int(R_hi, k2, k4, dk_over_2pi2, PL)
    crossed = (s_lo >= 1.0) & (s_hi <= 1.0)

    # Log-log secant estimate of the sigma2(R)=1 crossing, cheaper and
    # closer than the bracket's raw geometric midpoint.
    log_s_lo = jnp.log(jnp.clip(s_lo, 1e-300, None))
    log_s_hi = jnp.log(jnp.clip(s_hi, 1e-300, None))
    frac = jnp.clip((0.0 - log_s_lo) / (log_s_hi - log_s_lo), 0.0, 1.0)
    R0 = jnp.exp(jnp.log(R_lo) + frac * (jnp.log(R_hi) - jnp.log(R_lo)))

    def step(R, R_lo, R_hi):
        """One safeguarded Newton step: (R, R_lo, R_hi), each (N_cosmo,) -> the updated triple."""
        sigma2, k4int = _sigma2_k4int(R, k2, k4, dk_over_2pi2, PL)
        f = sigma2 - 1.0
        fprime = -2.0 * R * k4int
        R_newton = R - f / jnp.where(fprime < 0.0, fprime, -1e-300)
        # A few ULPs of slack on the bracket check: a converged step at a bracket edge must not be
        # mistaken for one outside it, whose bisection fallback can jump to a stale, distant R_hi.
        tol = 4.0 * jnp.finfo(R.dtype).eps * jnp.abs(R)
        in_bracket = (R_newton >= R_lo - tol) & (R_newton <= R_hi + tol)
        R_new = jnp.where(in_bracket, R_newton, jnp.sqrt(R_lo * R_hi))
        R_lo = jnp.where(f >= 0.0, jnp.maximum(R_lo, R), R_lo)
        R_hi = jnp.where(f < 0.0, jnp.minimum(R_hi, R), R_hi)
        return R_new, R_lo, R_hi

    R_final = R0
    R_prev = R0
    for _ in range(n_iter):
        R_prev = R_final
        R_final, R_lo, R_hi = step(R_final, R_lo, R_hi)
    resid = jnp.abs(R_final - R_prev) / R_final
    _, k4int_final = _sigma2_k4int(R_final, k2, k4, dk_over_2pi2, PL)

    knl = 1.0 / R_final
    neff = -3.0 + 2.0 * R_final**2 * k4int_final
    return knl, neff, crossed, resid
