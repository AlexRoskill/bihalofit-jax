"""
BiHalofit bispectrum (Takahashi et al. 2020, arXiv:1911.07886) with a
selectable linear P(k) backend.

Vectorisation, all inside the JIT'd kernel: N_cosmo batched, N_tri
broadcast, N_z vmapped. Units: k in h/Mpc, B in (Mpc/h)^6.
"""

from functools import partial
from typing import Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from ._config import resolve_dtype
from ._cosmology import as_cosmology
from .background import _E2_of_a
from ._knl_solver import _get_solver, _resolve_N_R, _solve_knl_neff
from ._pk_backends import (
    _pk_backend,
    _pk_log_interp_indices,
    _validate_baccoemu_bounds,
)
from ._validation import check_pk_options, check_solver
from ._utils import _smooth_min


# ---------------------------------------------------------------------------
# The kernel's cosmology-independent input, built once per make_* call
# ---------------------------------------------------------------------------


def _triangle_shape_params(k1, k2, k3):
    """BiHalofit's triangle shape parameters r1 = kmin/kmax, r2 = (kmid+kmin-kmax)/kmax.

    Parameters
    ----------
    k1, k2, k3 : (N_tri,) array-like
        Triangle sides, in any order and any common unit (or multipoles).

    Returns
    -------
    r1, r2 : (1, N_tri) ndarray
        Leading singleton axis, the convention the kernel expects.
    """
    ksorted = np.sort(
        np.stack([np.asarray(k1), np.asarray(k2), np.asarray(k3)], axis=0), axis=0
    )
    kmin, kmid, kmax = ksorted[0], ksorted[1], ksorted[2]
    r1 = (kmin / kmax)[np.newaxis, :]
    r2 = ((kmid + kmin - kmax) / kmax)[np.newaxis, :]
    return r1, r2


class TriangleSetup(NamedTuple):
    """The triangles prepared for :func:`bispectrum_all_z`, already on device.

    Built once by the ``make_*`` builders from the user's side arrays: the sides, BiHalofit's
    shape ratios r1/r2, and each side's index and weight on the P(k) grid. A pytree, so it
    passes through jit as one argument and its vmap in_axes is itself a TriangleSetup rather
    than a run of positional entries -- which is what lets the Limber kernel say "r1/r2 are
    shared, the rest vary per chi-node" by field name instead of by counting tuple positions.

    For fixed 3D triangles k1/k2/k3 and r1/r2 are (1, N_tri) and the indices and weights
    (N_tri,). For the Limber projection (``projected_bispectrum._limber_triangle_setup``) every
    field except r1/r2 carries a leading chi-node axis.
    """

    k1: Any
    k2: Any
    k3: Any
    r1: Any
    r2: Any
    idx_k1: Any
    w_k1: Any
    idx_k2: Any
    w_k2: Any
    idx_k3: Any
    w_k3: Any

    @classmethod
    def from_triangles(cls, k1, k2, k3, jdtype, k_lin=None):
        """A fixed set of 3D triangles, from side wavenumbers in h/Mpc.

        Parameters
        ----------
        k1, k2, k3 : (N_tri,) array-like
            Triangle sides in h/Mpc.
        jdtype : JAX dtype
        k_lin : (N_k,) array-like, optional
            P(k) grid the interpolation indices are built against (default ``_STANDARD_K_GRID``).

        Returns
        -------
        setup : TriangleSetup
            k1/k2/k3 and r1/r2 are (1, N_tri); indices and weights (N_tri,).
        """
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k3 = np.asarray(k3, dtype=float)
        r1, r2 = _triangle_shape_params(k1, k2, k3)
        return cls.build(
            k1[None, :], k2[None, :], k3[None, :], r1, r2, jdtype, k_lin=k_lin
        )

    @classmethod
    def build(cls, k1, k2, k3, r1, r2, jdtype, k_lin=None):
        """Interpolation indices per side, then one transfer to device.

        Parameters
        ----------
        k1, k2, k3 : (..., 1, N_tri) ndarray
            Sides in h/Mpc, as stored; the singleton axis broadcasts against N_cosmo in the kernel.
        r1, r2 : (1, N_tri) ndarray
            Shape parameters.
        jdtype : JAX dtype
        k_lin : (N_k,) array-like, optional

        Returns
        -------
        setup : TriangleSetup
            On device, floats in ``jdtype`` and indices int32; indices and weights (..., N_tri).
        """
        idx1, w1 = _pk_log_interp_indices(k1[..., 0, :], jdtype, k_lin=k_lin)
        idx2, w2 = _pk_log_interp_indices(k2[..., 0, :], jdtype, k_lin=k_lin)
        idx3, w3 = _pk_log_interp_indices(k3[..., 0, :], jdtype, k_lin=k_lin)
        npdtype = np.float64 if jdtype == jnp.float64 else np.float32
        f = lambda v: np.asarray(v, dtype=npdtype)
        i = lambda v: np.asarray(v, dtype=np.int32)
        # One device_put over the whole pytree rather than eleven separate
        # transfers: same result, measurably fewer dispatches.
        return jax.device_put(
            cls(
                k1=f(k1),
                k2=f(k2),
                k3=f(k3),
                r1=f(r1),
                r2=f(r2),
                idx_k1=i(idx1),
                w_k1=f(w1),
                idx_k2=i(idx2),
                w_k2=f(w2),
                idx_k3=i(idx3),
                w_k3=f(w3),
            )
        )


def _baryon_ratio(k1, k2, k3, a):
    """Bispectrum ratio with/without baryons (Takahashi+20, App. C, Eq. C1).

    k1, k2, k3 : triangle wavenumbers in h/Mpc (any broadcastable shape).
    a          : scalar scale factor for this redshift slice.
    Returns the multiplicative correction Rb, same broadcast shape as
    k1*k2*k3, with Rb=1 for z>5 (fit not calibrated beyond that).
    """
    z = 1.0 / a - 1.0

    # (a-0.5)**0.47 / (a-0.2)**1.41 have non-integer exponents, so clamp the
    # base to a safe value where the branch is inactive (avoids NaN gradients
    # through jnp.where's untaken branch).
    A0 = jnp.where(a > 0.5, 0.068 * jnp.where(a > 0.5, a - 0.5, 1.0) ** 0.47, 0.0)
    mu0 = 0.018 * a + 0.837 * a * a
    sigma0 = 0.881 * mu0
    alpha0 = 2.346

    A1 = jnp.where(a > 0.2, 1.052 * jnp.where(a > 0.2, a - 0.2, 1.0) ** 1.41, 0.0)
    mu1 = jnp.abs(0.172 + 3.048 * a - 0.675 * a * a)
    sigma1 = (0.494 - 0.039 * a) * mu1

    ks = 29.90 - 38.73 * a + 24.30 * a * a
    alpha2 = 2.25
    beta2 = 0.563 / ((a / 0.060) ** 0.02 + 1.0) / alpha2

    def _leg(k):
        """Single-side factor of Rb at wavenumber k (h/Mpc), same shape as k."""
        xi = jnp.log10(k)
        return (
            A0 * jnp.exp(-((jnp.abs(xi - mu0) / sigma0) ** alpha0))
            - A1 * jnp.exp(-((jnp.abs(xi - mu1) / sigma1) ** 2))
            + (1.0 + (k / ks) ** alpha2) ** beta2
        )

    Rb = _leg(k1) * _leg(k2) * _leg(k3)
    return jnp.where(z > 5.0, 1.0, Rb)


def _interp_pk_sides(PL, setup):
    """Linear P(k) at each triangle side.

    Log-linear on the P(k) grid, with the index/weight pairs precomputed on
    the host (see TriangleSetup.build); gradients flow through log_PL.

    Parameters
    ----------
    PL : (N_cosmo, N_k)
        Linear P(k) in (Mpc/h)^3 on the grid the setup was built against.
    setup : TriangleSetup

    Returns
    -------
    PL1, PL2, PL3 : (N_cosmo, N_tri)
    """
    log_PL = jnp.log(PL)  # (N_cosmo, N_k)

    def at(idx, w):
        """P(k) at grid index idx with right-neighbour weight w: (N_tri,) -> (N_cosmo, N_tri)."""
        return jnp.exp(log_PL[:, idx] * (1.0 - w) + log_PL[:, idx + 1] * w)

    return (
        at(setup.idx_k1, setup.w_k1),
        at(setup.idx_k2, setup.w_k2),
        at(setup.idx_k3, setup.w_k3),
    )


def _F2(ka, kb, kc):
    """Second-order PT mode-coupling kernel for the (ka, kb) pair of a closed triangle with third side kc.

    Takes three broadcastable arrays of side lengths (any common unit) and returns F2, dimensionless,
    in their broadcast shape.
    """
    cos12 = (kc**2 - ka**2 - kb**2) / (2.0 * ka * kb)
    return 5.0 / 7.0 + 0.5 * cos12 * (ka / kb + kb / ka) + 2.0 / 7.0 * cos12**2


def Btree(PL1, PL2, PL3, k1, k2, k3):
    """Tree-level (2nd-order PT) bispectrum.

    PL1, PL2, PL3 : linear P(k) at each triangle side
    k1, k2, k3    : triangle wavenumbers in h/Mpc
    No dn correction and no k_NL/n_eff dependence, so this term needs no
    Newton solve.

    Returns B in (Mpc/h)^6, broadcast shape of the inputs.
    """
    return 2.0 * (
        _F2(k1, k2, k3) * PL1 * PL2
        + _F2(k2, k3, k1) * PL2 * PL3
        + _F2(k3, k1, k2) * PL3 * PL1
    )


def B1h(logsigma8z, neff, ns, r1, r2, q1, q2, q3):
    """1-halo bispectrum term (Takahashi+20).

    logsigma8z : log10(sigma8(z)) (scalar or array)
    neff       : effective spectral index at k_NL (scalar or array)
    ns         : primordial spectral index
    r1, r2     : triangle shape parameters (scalar or array)
    q1, q2, q3 : k_i / k_NL (broadcastable shape)

    Returns B_1h in (Mpc/h)^6, broadcast shape of the inputs.
    """
    gamman = 10.0 ** (0.182 + 0.57 * neff)
    an = 10.0 ** (
        -2.167
        - 2.944 * logsigma8z
        - 1.106 * logsigma8z**2
        - 2.865 * logsigma8z**3
        - 0.310 * r1**gamman
    )
    bn = 10.0 ** (
        -3.428 - 2.681 * logsigma8z + 1.624 * logsigma8z**2 - 0.095 * logsigma8z**3
    )
    cn = 10.0 ** (0.159 - 1.107 * neff)
    alphan = _smooth_min(
        10.0
        ** (
            -4.348
            - 3.006 * neff
            - 0.5745 * neff**2
            + 10.0 ** (-0.9 + 0.2 * neff) * r2**2
        ),
        1.0 - 2.0 / 3.0 * ns,
        1e-3,
    )  # smooth min, so d(alphan)/d(neff) is continuous
    betan = 10.0 ** (
        -1.731 - 2.845 * neff - 1.4995 * neff**2 - 0.2811 * neff**3 + 0.007 * r2
    )

    def T(q):
        """Per-side 1-halo factor at q = k/k_NL."""
        cnq = cn * q
        return cnq / ((an * q**alphan + bn * q**betan) * (1.0 + cnq))

    return T(q1) * T(q2) * T(q3)


def B3h(logsigma8z, neff, Omegamz, q1, q2, q3, PL1, PL2, PL3, k1, k2, k3):
    """3-halo bispectrum term (Takahashi+20).

    logsigma8z : log10(sigma8(z))
    neff       : effective spectral index at k_NL
    Omegamz    : Omega_m(z)
    q1, q2, q3 : k_i / k_NL
    PL1, PL2, PL3 : linear P(k) at each triangle side
    k1, k2, k3 : triangle wavenumbers in h/Mpc

    Returns B_3h in (Mpc/h)^6, broadcast shape of the inputs.
    """
    fn = 10.0 ** (-10.533 - 16.838 * neff - 9.3048 * neff**2 - 1.8263 * neff**3)
    gn = 10.0 ** (2.787 + 2.405 * neff + 0.4577 * neff**2)
    hn = 10.0 ** (-1.118 - 0.394 * neff)
    mn = 10.0 ** (-2.605 - 2.434 * logsigma8z + 5.71 * logsigma8z**2)
    nn = 10.0 ** (-4.468 - 3.08 * logsigma8z + 1.035 * logsigma8z**2)
    mun = 10.0 ** (15.312 + 22.977 * neff + 10.9579 * neff**2 + 1.6586 * neff**3)
    nun = 10.0 ** (1.347 + 1.246 * neff + 0.4525 * neff**2)
    pn = 10.0 ** (0.071 - 0.433 * neff)
    dn = 10.0 ** (-0.483 + 0.892 * logsigma8z - 0.086 * Omegamz)
    en = 10.0 ** (-0.632 + 0.646 * neff)

    def I_func(q):
        """Per-side damping factor at q = k/k_NL."""
        return 1.0 / (1.0 + en * q)

    def PE(q, Pk):
        """Enhanced power at q = k/k_NL, from the linear Pk at that side."""
        pq3 = (pn * q) ** 3
        return (1.0 + fn * q**2) / (1.0 + gn * q + hn * q**2) * Pk + pq3 / (
            mn * q**mun + nn * q**nun
        ) / (1.0 + pq3)

    I_prod = I_func(q1) * I_func(q2) * I_func(q3)
    pe1 = PE(q1, PL1)
    pe2 = PE(q2, PL2)
    pe3 = PE(q3, PL3)

    return (
        2.0
        * I_prod
        * (
            (_F2(k1, k2, k3) + dn * q3) * pe1 * pe2
            + (_F2(k2, k3, k1) + dn * q1) * pe2 * pe3
            + (_F2(k3, k1, k2) + dn * q2) * pe3 * pe1
        )
    )


@partial(
    jax.jit,
    static_argnames=(
        "pk_source",
        "R_min",
        "R_max",
        "n_iter",
        "jdtype",
        "term",
        "baryonic_boost",
        "k_lin",
    ),
)
def bispectrum_all_z(
    cosmo,
    a,
    setup,
    pk_source,
    R_min,
    R_max,
    n_iter,
    jdtype=jnp.float64,
    term="full",
    baryonic_boost=False,
    k_lin=None,
):
    """Linear P(k) + the BiHalofit kernel, vmapped over the `a` axis, in one
    jax.jit -- a warm call is a single XLA launch.

    cosmo is a Cosmology of (N_cosmo,) arrays (see Cosmology.as_arrays) and
    setup a TriangleSetup; both are pytrees, so each passes as one argument and each
    has a structured vmap in_axes rather than a run of positional entries.

    -> (B, resid): B is (N_cosmo, N_z, N_tri); resid is (N_cosmo, N_z), the
    Newton residual per cosmology/slice, for make_bihalofit's tol check.

    The `a` axis is whatever the caller is batching over. make_bihalofit
    passes a redshift grid with one fixed triangle set. The Limber kernel
    vmaps this whole function over chi-nodes instead, passing a length-1 `a`
    and that node's own k = ell/chi -- measured free, since the emulator call
    is launch-bound rather than row-bound at these sizes.

    `a` need not be sorted: every backend computes the sigma8-normalisation
    slice explicitly at a=1, and the per-slice solve has no dependency on
    neighbouring slices, so output order matches input order.

    Everything except the static args is dynamic, so one executable serves any
    N_z/N_tri until the shape itself changes.

    k_lin, if given, is a hashable tuple (static, like every other argument
    here -- a plain array can't be a jit static_argname) of a custom P(k)
    grid in h/Mpc, forwarded to both the k_NL/n_eff solver and the P(k)
    backend; None uses the built-in _STANDARD_K_GRID. make_bihalofit/
    make_projected_bihalofit accept k_lin as an ordinary array and convert
    it to a tuple before it reaches here.

    Parameters
    ----------
    cosmo : Cosmology
        Fields are (N_cosmo,) arrays.
    a : (N_z,)
        Scale factors.
    setup : TriangleSetup
    pk_source : str
        Linear P(k) backend.
    R_min, R_max : float
        k_NL root-search bracket in Mpc/h.
    n_iter : int
        Newton iterations.
    jdtype : JAX dtype, optional
    term : {'full', '1h', '3h', 'tree'}, optional
    baryonic_boost : bool, optional
        Multiply by the Takahashi+20 baryon ratio.
    k_lin : tuple of float or None, optional

    Returns
    -------
    B : (N_cosmo, N_z, N_tri)
        Bispectrum in (Mpc/h)^6; NaN where sigma^2(R) never reaches 1.
    resid : (N_cosmo, N_z)
        Newton residual per cosmology and slice.
    """
    k_lin_arr = None if k_lin is None else np.asarray(k_lin)
    k2_j, k4_j, dk_j, w8_j = _get_solver(k_lin_arr, jdtype)

    pk_fn = _pk_backend(pk_source, k_lin=k_lin_arr)
    # The backend takes its own argument order; name them rather than splat.
    PL_all, PL_z0 = pk_fn(
        cosmo.omega_cold,
        cosmo.omega_baryon,
        cosmo.hubble,
        cosmo.ns,
        cosmo.sigma8_cold,
        cosmo.neutrino_mass,
        cosmo.w0,
        cosmo.wa,
        a,
        jdtype,
    )
    # (N_z,N_cosmo,N_k),(N_cosmo,N_k)

    sigma8_sq_z0 = jnp.sum(
        PL_z0 * w8_j, axis=1
    )  # top-hat sigma8^2 of the z = 0 spectrum, shared by the slices

    def _slice(PL, a):
        """One slice of `a`: the BiHalofit fitting formula at fixed geometry.

        Closes over cosmo/setup/PL_z0 and the solver constants, all shared
        across the batch.

        Parameters
        ----------
        PL : (N_cosmo, N_k)
            Linear P(k) at this slice.
        a : scalar
            Scale factor of this slice.

        Returns
        -------
        B : (N_cosmo, N_tri)
        resid : (N_cosmo,)
        """
        k1, k2, k3 = setup.k1, setup.k2, setup.k3  # used throughout below

        # term is static, so each branch traces only what it needs: 'tree' never
        # traces the Newton solve, '1h' never traces the B3h coefficients.
        if term == "tree":
            B = Btree(*_interp_pk_sides(PL, setup), k1, k2, k3)
            if baryonic_boost:
                B = B * _baryon_ratio(k1, k2, k3, a)
            # No Newton solve, so a zero residual: 'tree' always passes a tol check.
            resid = jnp.zeros((PL.shape[0],), dtype=PL.dtype)
            return B, resid

        # --- k_NL and n_eff, from sigma^2(R_NL) = 1 ---
        # Where sigma^2 never reaches 1 on [R_min, R_max], `crossed` is False and B is set to NaN
        # below. No jax.debug.callback: a host callback per launch costs ~100x the kernel.
        knl, neff, crossed, resid = _solve_knl_neff(
            PL, k2_j, k4_j, dk_j, R_min, R_max, n_iter
        )
        # Diagnostic only: never differentiate through it, never let it affect B.
        resid = jax.lax.stop_gradient(resid)

        # --- log sigma8(z) ---
        # sigma8(z) = sigma8 * (top-hat sigma8 of this slice / that of z = 0): D(z) for a spectrum rescaled in time,
        # and the growth of the spectrum itself otherwise (the input sigma8 is kept, as in the original code).
        growth = jnp.sqrt(jnp.sum(PL * w8_j, axis=1) / sigma8_sq_z0)
        logsigma8z = jnp.log10(cosmo.sigma8_cold * growth)

        # --- Omega_m(z) ---
        # omega_cold already includes baryons; Ω_ν = Σm_ν / (93.14 eV h²).
        omega_m = cosmo.omega_cold + cosmo.neutrino_mass / (93.14 * cosmo.hubble**2)
        E2 = _E2_of_a(a, omega_m, cosmo.w0, cosmo.wa)
        Omegamz = omega_m * a**-3 / E2

        # broadcast to (N_cosmo, N_tri)
        knl_ = knl[:, None]
        neff_ = neff[:, None]
        ls8z_ = logsigma8z[:, None]
        Omz_ = Omegamz[:, None]
        q1 = k1 / knl_
        q2 = k2 / knl_
        q3 = k3 / knl_

        # PRECISION: every coefficient below is 10**(c*neff_ + ...) with c up to
        # ~23. d(10^x)/dx = ln(10)*10^x, so an absolute perturbation in neff_ is
        # amplified by c in the exponent -- a ULP-level change in the k_NL/n_eff
        # solve shows up as ~1e-8..1e-14 relative in B. Check any change to that
        # solve against this amplification rather than assuming it is benign.
        if term in ("1h", "full"):
            B_1h = B1h(
                ls8z_, neff_, cosmo.ns[:, None], setup.r1, setup.r2, q1, q2, q3
            )  # (N_cosmo, N_tri)

        if term in ("3h", "full"):
            B_3h = B3h(
                ls8z_, neff_, Omz_, q1, q2, q3, *_interp_pk_sides(PL, setup), k1, k2, k3
            )

        if term == "1h":
            B = B_1h
        elif term == "3h":
            B = B_3h
        else:
            B = B_1h + B_3h  # (N_cosmo, N_tri)

        if baryonic_boost:
            B = B * _baryon_ratio(k1, k2, k3, a)

        # NaN-poison cosmologies where sigma^2(R) never reached 1. The false
        # branch is a literal, so gradients of valid cosmologies are unaffected.
        B = jnp.where(crossed[:, None], B, jnp.nan)
        return B, resid

    # PL and a vary per slice; everything else is closed over.
    B_over_z, resid_over_z = jax.vmap(_slice)(PL_all, a)
    # (N_z,N_cosmo,N_tri), (N_z,N_cosmo)
    B_final = jnp.moveaxis(B_over_z, 0, 1)  # (N_cosmo, N_z, N_tri)
    resid_final = jnp.moveaxis(resid_over_z, 0, 1)  # (N_cosmo, N_z)
    return B_final, resid_final


def make_bihalofit(
    k1,
    k2,
    k3,
    expfactors,
    pk_source="syren",
    N_R=15,
    tol=None,
    R_min=1e-6,
    R_max=40.0,
    dtype=np.float32,
    validate=False,
    term="full",
    baryonic_boost=False,
    k_lin=None,
):
    r"""Build a BiHalofit callable for fixed triangles and redshifts.

    Set-up and JIT compilation happen once; call the result at many cosmologies.

    Parameters
    ----------
    k1, k2, k3 : array-like, shape (N_tri,)
        Triangle sides in h/Mpc.
    expfactors : array-like, shape (N_z,) or scalar
        Scale factors :math:`a = 1/(1+z)`.
    pk_source : {'syren', 'baccoemu', 'eh', 'eh_99'}, optional
        Linear P(k) backend (default 'syren'). 'baccoemu' needs the optional ``baccoemu`` package.
    N_R : int, optional
        Newton iterations for :math:`\sigma^2(R_\mathrm{NL})=1` (default 15; ~8 suffice).
        Exclusive with ``tol``.
    tol : float, optional
        Target relative error on :math:`k_\mathrm{NL}` and :math:`n_\mathrm{eff}`, converted to an
        iteration count before tracing. The residual is then checked (on concrete values only);
        a miss is handled as ``validate`` says and the error names ``N_R`` as the fix.
    R_min, R_max : float, optional
        Root-search bracket in Mpc/h (default 1e-6 to 40); must be positive.
    dtype : numpy dtype, optional
        ``np.float32`` (default; within 0.02% of float64 and ~7x faster on the GPU) or
        ``np.float64``, which needs :func:`~bihalofit_jax.enable_x64`. Use float64 for gradients
        away from ``neutrino_mass=0``.
    validate : bool, optional
        If :math:`\sigma^2` never reaches 1 or ``tol`` is missed: raise (True) or return NaN for
        those entries (False, default), which an MCMC can reject.
    term : {'full', '1h', '3h', 'tree'}, optional
        Default 'full' (1h + 3h). ``'tree'`` is tree-level perturbation theory and never NaN.
    baryonic_boost : bool, optional
        Multiply by the baryon-feedback ratio of Takahashi et al. 2020, Appendix C.
    k_lin : array-like, optional
        Linear P(k) grid in h/Mpc, replacing the 600-point ``_STANDARD_K_GRID``. Its log-linear
        interpolation sets the accuracy floor of :math:`B` (error :math:`\propto 1/N_k^2`). Not
        for ``'baccoemu'``, whose grid is fixed by the emulator.

    Returns
    -------
    bihalofit : callable
        ``bihalofit(cosmo)`` -> (N_cosmo, N_z, N_tri), for a :class:`~bihalofit_jax.Cosmology`.
        Differentiable in every parameter; ``jax.grad`` returns a ``Cosmology`` of derivatives.
        Pure (composes with ``jit``, ``vmap``, ``scan``) unless ``tol`` or ``validate`` is set,
        which syncs to the host. The baccoemu bounds check is skipped under ``jit``.

    Notes
    -----
    :math:`k_\mathrm{NL} = 1 / R_\mathrm{NL}` solves

    .. math::
        \sigma^2(R_\mathrm{NL}) = 1, \qquad
        \sigma^2(R) = \frac{1}{2\pi^2} \int dk\, k^2 P_L(k)\, e^{-k^2 R^2}

    with effective spectral index

    .. math::
        n_\mathrm{eff} = -3 + 2 R_\mathrm{NL}^2
            \left.\frac{1}{2\pi^2} \int dk\, k^4 P_L(k)\, e^{-k^2 R^2}
            \right|_{R = R_\mathrm{NL}}

    and :math:`B = B_\mathrm{1h} + B_\mathrm{3h}` (Takahashi et al. 2020).
    """
    check_pk_options(pk_source, term, k_lin)

    N_R = _resolve_N_R(N_R, tol)
    jdtype = resolve_dtype(dtype)
    expfactors = np.atleast_1d(np.asarray(expfactors, dtype=float))
    a_j = jnp.asarray(expfactors, dtype=jdtype)
    # TriangleSetup takes k_lin as an array; the jitted kernel needs it hashable, as a tuple.
    k_lin_arr = None if k_lin is None else np.asarray(k_lin, dtype=float)
    k_lin_tup = None if k_lin_arr is None else tuple(k_lin_arr.tolist())
    setup = TriangleSetup.from_triangles(k1, k2, k3, jdtype, k_lin=k_lin_arr)

    kernel = partial(
        bispectrum_all_z,
        pk_source=pk_source,
        R_min=R_min,
        R_max=R_max,
        jdtype=jdtype,
        term=term,
        baryonic_boost=baryonic_boost,
        k_lin=k_lin_tup,
    )

    # The bounds check needs concrete values, so it is skipped under tracing:
    # a jitted call is unchecked. It runs before any H2D transfer so that
    # validating never forces a pull back from the GPU.
    def _check(cosmo):
        """Raise if a baccoemu input is outside the emulator bounds; returns None."""
        if pk_source == "baccoemu":
            _validate_baccoemu_bounds(cosmo._asdict(), expfactors)

    if tol is None and not validate:
        # Pure path: no host sync, so this composes under jit/vmap/scan.
        def bihalofit(cosmo):
            """Cosmology (or dict of its fields) -> B, (N_cosmo, N_z, N_tri) in (Mpc/h)^6."""
            cosmo = as_cosmology(cosmo)
            _check(cosmo)
            return kernel(cosmo.as_arrays(jdtype), a_j, setup, n_iter=N_R)[0]

        return bihalofit

    # Checked path: tol/validate pull B and the residual to the host, so it is not jittable.
    def bihalofit(cosmo):
        """Cosmology (or dict of its fields) -> B, (N_cosmo, N_z, N_tri) in (Mpc/h)^6, solver-checked."""
        cosmo = as_cosmology(cosmo)
        _check(cosmo)
        B_out, resid_out = kernel(cosmo.as_arrays(jdtype), a_j, setup, n_iter=N_R)
        return check_solver(
            B_out,
            resid_out,
            tol=tol,
            validate=validate,
            N_R=N_R,
            jdtype=jdtype,
            over="cosmology/redshift",
        )

    return bihalofit
