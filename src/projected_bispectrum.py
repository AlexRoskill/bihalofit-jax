"""
Limber-projected bispectrum, with an independent tracer on each triangle leg.

    B(l1, l2, l3) = INT dchi  W1(chi) W2(chi) W3(chi) / chi^4
                    * B_delta((l1+1/2)/chi, (l2+1/2)/chi, (l3+1/2)/chi; z(chi))

The +1/2 is the LoVerde & Afshordi (2008) extended-Limber shift (see
_limber_triangle_setup): it cuts the Limber error from O(1/l) to O(1/l^2).

Each leg picks its own projection kernel W -- 'lensing', 'counts',
'cmb_lensing' or 'ia' -- so one function covers cosmic shear, galaxy-galaxy lensing,
clustering, CMB-lensing cross bispectra and the intrinsic-alignment terms. 'lensing' on all three legs with
a single source distribution gives the usual cosmic-shear B_kappa.
B_delta is BiHalofit on a linear P(k) backend; see make_projected_bihalofit
for the kernel definitions and every parameter.

Density parameters are dimensionless, not physical densities: omega_cold is
Omega_cdm + Omega_b and omega_baryon is Omega_b. Do not divide by h^2.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

# np.trapezoid added in NumPy 2.0; np.trapz removed in 2.0.
try:
    _np_trapezoid = np.trapezoid
except AttributeError:
    _np_trapezoid = np.trapz

# Private internals shared with bihalofit_jax
from .bispectrum import (
    bispectrum_all_z,
)  # P(k) + BiHalofit kernel; vmapped over chi-node below
from .bispectrum import TriangleSetup, _triangle_shape_params
from ._validation import check_pk_options, check_solver
from ._knl_solver import (
    _resolve_N_R,  # N_R/tol mutual-exclusivity guard + resolution (static, pre-trace)
)
from ._config import resolve_dtype
from ._diagnostics import _DiagnosticInputs, warn_if_geometry_inadequate
from .tracers import make_tracer_kernel, validate_tracer_config, _IA_Z_PIVOT_DEFAULT
from .background import (
    _comoving_distance,
    _smooth_lookup,
    _background_zmax,
    _background_zgrid,
    growth_ratios,  # D(a)/D(1), for the alignment amplitude 1/D(z)
    _GROWTH_ODE_A0,
)
from ._cosmology import Cosmology, IA, as_cosmology, as_ia
from ._utils import (
    gauss_legendre,
    _as_jax_vector,  # coerce a scalar/array param to a JAX (N,) vector
)
from ._pk_backends import _validate_baccoemu_bounds

# Scale factors below this are clipped before the growth solve (its ODE starts at
# _GROWTH_ODE_A0): deep or extrapolated chi-nodes carry no weight, but a NaN there
# would still reach the gradient through the node-dropping `where`.
_IA_A_FLOOR = 2.0 * _GROWTH_ODE_A0

# Default χ-integration bounds in Mpc/h.
_CHI_MIN_DEFAULT = 10.0  # Mpc/h  (z ≈ 0.003)
_CHI_MAX_DEFAULT = 5800.0  # Mpc/h  (z ≈ 6 for Ω_m ~ 0.3)

# A chi-node whose weight is below this fraction of the largest is dropped (see the kernel).
_NODE_WEIGHT_FLOOR = 1e-9


# ---------------------------------------------------------------------------
# Source-distribution / quadrature utilities  (plain numpy; concrete inputs)
# ---------------------------------------------------------------------------


def smail_nz(z, z0=0.3, alpha=2.0, beta=1.0):
    r"""
    Smail-type source redshift distribution, normalized to unit integral.

    .. math::
        n(z) \propto (z/z_0)^\alpha \exp\!\left(-(z/z_0)^\beta\right),
        \qquad \int n(z)\, dz = 1

    Default parameters (z0=0.3, alpha=2, beta=1) give median redshift ≈ 0.9,
    representative of an LSST DESC SRD Y10 source sample.

    Parameters
    ----------
    z : (N_z,) array-like
        Redshift grid; the normalisation is a sum over ``np.gradient(z)``.
    z0, alpha, beta : float, optional
        Shape parameters.

    Returns
    -------
    nz : (N_z,) ndarray
        Zero at z <= 0.
    """
    z = np.asarray(z, dtype=float)
    x = np.where(z > 0, z / z0, 0.0)
    n = np.where(z > 0, x**alpha * np.exp(-(x**beta)), 0.0)
    dz = np.gradient(z)
    return n / np.sum(n * dz)


# ---------------------------------------------------------------------------
# Single-cosmology Limber kernel (pure JAX, differentiable in all params)
# ---------------------------------------------------------------------------


def _limber_triangle_setup(
    ell1, ell2, ell3, chi_nodes, jdtype, k_lin=None, limber_shift=0.5
):
    """Per-chi-node triangles, k_i = (ell_i + limber_shift) / chi.

    ``limber_shift=0.5`` (default) is the LoVerde & Afshordi (2008)
    extended-Limber shift: it reconciles the discrete spherical-multipole
    index ell with the continuous flat-sky wavenumber, cutting the Limber
    error from O(1/ell) to O(1/ell^2).

    ``limber_shift=0.0`` gives the plain k = ell/chi convention. That is
    the less accurate choice, but it is what several other codes use --
    `threepoint` among them (`bispectrum.cpp`, `bispec(ell1/f_K, ...)`) --
    so it is exposed for like-for-like comparison. The difference is
    O(1/ell) and so matters most at large apertures: on B_kappa it runs
    from a few percent at ell~50 to ~0.02% at ell~2e4.

    The shift is applied before r1/r2 too, so the shifted triangle is used
    consistently everywhere below. r1/r2 carry no chi-node axis: they are
    scale-invariant ratios, so the uniform 1/chi rescaling of all three
    sides cancels out of them exactly.

    Parameters
    ----------
    ell1, ell2, ell3 : (N_ltri,) ndarray
        Multipole triangle sides.
    chi_nodes : (N_chi,) ndarray
        Comoving distances in Mpc/h.
    jdtype : JAX dtype
    k_lin : (N_k,) array-like, optional
        P(k) grid (default ``_STANDARD_K_GRID``).
    limber_shift : float, optional
        Added to each ell before dividing by chi (default 0.5).

    Returns
    -------
    setup : TriangleSetup
        k1/k2/k3 are (N_chi, 1, N_ltri), indices and weights (N_chi, N_ltri), r1/r2 (1, N_ltri).
    """
    ell1 = ell1 + limber_shift
    ell2 = ell2 + limber_shift
    ell3 = ell3 + limber_shift
    r1, r2 = _triangle_shape_params(ell1, ell2, ell3)
    per_node = lambda ell: (ell[np.newaxis, :] / chi_nodes[:, np.newaxis])[
        :, np.newaxis, :
    ]
    k1, k2, k3 = per_node(ell1), per_node(ell2), per_node(ell3)
    return TriangleSetup.build(k1, k2, k3, r1, r2, jdtype, k_lin=k_lin)


def _build_limber_kernel(
    ell1,
    ell2,
    ell3,
    dndz,
    tracers,
    z_source,
    z_bg,
    chi_nodes,
    gl_weights,
    pk_source,
    N_R,
    R_min,
    R_max,
    jdtype,
    term="full",
    baryonic_boost=False,
    k_lin=None,
    limber_shift=0.5,
    z_pivot=_IA_Z_PIVOT_DEFAULT,
):
    r"""Assemble and return a pure-JAX single-cosmology kernel.

    The returned function takes the eight scalar cosmological parameters
    (omega_cold, omega_baryon, sigma8_cold, ns, neutrino_mass, w0, wa, hubble)
    plus ``nz`` and ``bias``, length-3 tuples of per-leg n(z) values and
    clustering biases (``None`` for a leg that has none), and ``ia``, the
    alignment parameters ``(A_IA, eta)`` (``None`` when no leg is an 'ia' leg), and returns
    :math:`B` of shape (N_ltri,) plus the worst-chi-node k_NL/n_eff solve
    residual. ``nz``/``bias`` are traced; the z *grids* are not.  All other
    configuration (triangle grid, :math:`\chi`-nodes, per-leg z-grids/tracers,
    JIT'd bispectrum kernel) is baked in.

    ``dndz[i]`` is ``(z_arr, nz_arr)`` for ``tracers[i]`` in
    ``('lensing', 'counts', 'ia')``, else ``None``. ``z_bg`` is the shared
    background z-grid (see ``_background_zmax``) used to invert chi_nodes
    into z_nodes/a_nodes for ALL three legs, independent of any leg's own
    dndz.

    Parameters
    ----------
    ell1, ell2, ell3 : (N_ltri,) ndarray
        Multipole triangle sides.
    dndz : length-3 tuple
        Per-leg normalised ``(z_arr, nz_arr)`` or None.
    tracers : length-3 tuple of str
    z_source : length-3 tuple of float
        Source-plane redshift per leg (used for 'cmb_lensing').
    z_bg : (N_zbg,) ndarray
        Shared background z-grid.
    chi_nodes, gl_weights : (N_chi,) ndarray
        Limber quadrature nodes (Mpc/h) and weights.
    pk_source : str
    N_R : int
        Newton iterations.
    R_min, R_max : float
        k_NL root-search bracket in Mpc/h.
    jdtype : JAX dtype
    term, baryonic_boost, k_lin, limber_shift, z_pivot : optional
        As in :func:`make_projected_bihalofit`.

    Returns
    -------
    kernel : callable
        ``kernel(omega_cold, omega_baryon, sigma8_cold, ns, neutrino_mass, w0, wa, hubble, nz,
        bias, ia)`` -> ``(B, resid_max)``: B of shape (N_ltri,) and a scalar residual.
    """
    N_zbg = len(z_bg)

    setup_axes = TriangleSetup(*(0,) * len(TriangleSetup._fields))._replace(
        r1=None, r2=None
    )

    # _limber_triangle_setup takes k_lin as an array; the jitted kernel needs it hashable, as a tuple.
    k_lin_tup = (
        None if k_lin is None else tuple(np.asarray(k_lin, dtype=float).tolist())
    )

    setup_j = _limber_triangle_setup(
        ell1, ell2, ell3, chi_nodes, jdtype, k_lin=k_lin, limber_shift=limber_shift
    )

    # Concrete arrays baked into the closure.
    chi_nodes_j = jnp.asarray(chi_nodes, dtype=jdtype)
    gl_w_j = jnp.asarray(gl_weights, dtype=jdtype)
    z_bg_j = jnp.asarray(z_bg, dtype=jdtype)

    # One closure per leg, each binding its own static data. The kernel below
    # never branches on tracer type -- see tracers.py.
    tracer_kernels = [
        make_tracer_kernel(
            tracers[i], dndz[i], z_source[i], chi_nodes_j, z_bg, jdtype, z_pivot
        )
        for i in range(3)
    ]
    has_ia = "ia" in tracers

    def kernel(
        omega_cold,
        omega_baryon,
        sigma8_cold,
        ns,
        neutrino_mass,
        w0,
        wa,
        hubble,
        nz,
        bias,
        ia,
    ):
        """Single-cosmology projected bispectrum; see _build_limber_kernel for the arguments.

        Returns ``(B, resid_max)``: B (N_ltri,) and the worst active chi-node's Newton residual.
        """
        Omega_nu = neutrino_mass / 93.14 / hubble**2
        Omega_m = omega_cold + Omega_nu  # dimensionless Ω_m

        # Background z(chi)/a(chi), shared by all three legs (see _smooth_lookup
        # for why this needs a C^1 inversion rather than plain jnp.interp).
        chi_bg = _comoving_distance(z_bg_j, Omega_m, w0, wa)  # (N_zbg,)
        z_nodes = _smooth_lookup(chi_bg, z_bg_j, chi_nodes_j, N_zbg)  # (N_chi,)
        a_nodes = 1.0 / (1.0 + z_nodes)  # (N_chi,)

        # Growth factor at the chi-nodes, for the alignment amplitude 1/D(z). It depends on
        # the cosmology through a(chi), so it is computed here, and only if a leg needs it.
        if has_ia:
            D_nodes = growth_ratios(
                jnp.clip(a_nodes, _IA_A_FLOOR, 1.0), Omega_m, w0, wa, jdtype
            )
            ia_leg = (ia[0], ia[1], D_nodes)
        else:
            ia_leg = None

        W1, W2, W3 = (
            W(Omega_m, w0, wa, chi_bg, a_nodes, z_nodes, nz_i, b_i, ia_leg)
            for W, nz_i, b_i in zip(tracer_kernels, nz, bias)
        )
        weight = gl_w_j / chi_nodes_j**4 * W1 * W2 * W3  # (N_chi,)

        # Nodes far behind every source (weight below the floor, ~1e-11 at z~100) can make the k_NL
        # solve fail; they are evaluated at a=1 and weighted zero. A NaN weight counts as inactive.
        active = jnp.abs(weight) > _NODE_WEIGHT_FLOOR * jnp.nanmax(jnp.abs(weight))
        weight = jnp.where(active, weight, 0.0)
        a_eval = jnp.where(active, a_nodes, 1.0)

        s1 = lambda v: _as_jax_vector(v, jdtype)  # scalar -> (1,) JAX, N_cosmo=1 here

        # B_delta and the k_NL/n_eff residual at every chi-node x triangle, vmapped over nodes, each
        # with a length-1 `a` and its own k = (ell + shift)/chi. The ratios r1, r2 are the same at
        # every node; the cosmology is shared.
        B_all, resid_all = jax.vmap(
            partial(
                bispectrum_all_z,
                pk_source=pk_source,
                R_min=R_min,
                R_max=R_max,
                n_iter=N_R,
                jdtype=jdtype,
                term=term,
                baryonic_boost=baryonic_boost,
                k_lin=k_lin_tup,
            ),
            in_axes=(None, 0, setup_axes),  # cosmo, a, setup
        )(
            Cosmology(
                s1(omega_cold),
                s1(omega_baryon),
                s1(sigma8_cold),
                s1(ns),
                s1(hubble),
                s1(neutrino_mass),
                s1(w0),
                s1(wa),
            ),
            a_eval[:, None],
            setup_j,
        )  # (N_chi,1,1,N_ltri), (N_chi,1,1)

        # drop the degenerate N_cosmo=1 and N_z=1 axes; a dropped node contributes
        # exactly zero even if the solve is unconverged there (0 * NaN is NaN)
        B_leg = jnp.where(active[:, None], B_all[:, 0, 0, :], 0.0)
        resid_leg = resid_all[:, 0, 0]

        B_out = jnp.einsum("j,ji->i", weight, B_leg)  # (N_ltri,)
        resid_max = jnp.max(
            jnp.where(active, resid_leg, 0.0)
        )  # worst *active* chi-node
        return B_out, resid_max

    return kernel


def _check_nz_values(nz_arr, z_arr, name):
    """Raise unless n(z) is finite and non-negative with a positive integral.

    The tracer kernels divide by that integral in-trace, so a zero-integral
    n(z) would otherwise reach B and its gradients as inf/NaN. JAX tracers
    cannot be checked by value and pass through.

    Parameters
    ----------
    nz_arr : (..., N_z) array-like
        n(z) values (one row per cosmology if 2-D).
    z_arr : (N_z,) ndarray
        Redshift grid.
    name : str
        Argument name for the error message.

    Returns
    -------
    None
        Raises ValueError.
    """
    try:
        nz_arr = np.asarray(nz_arr, dtype=float)
    except jax.errors.TracerArrayConversionError:
        return
    norm = _np_trapezoid(nz_arr, z_arr)
    if not (
        np.all(np.isfinite(nz_arr))
        and np.all(nz_arr >= 0)
        and np.all(np.isfinite(norm))
        and np.all(norm > 0)
    ):
        raise ValueError(
            f"{name} must be finite and non-negative with a positive integral."
        )


def _resolve_nz(nz, nz_default, z_arrs, N_cosmo, jdtype):
    """Per-leg n(z) values as (N_cosmo, N_z), or None where a leg has none.

    ``nz`` is the caller's override (a length-3 tuple, entries None to keep
    the build-time default); a leg's values may be given as (N_z,) to share
    one distribution across the whole cosmology batch. Unnormalised is fine
    -- the tracer kernels normalise in-trace.

    Parameters
    ----------
    nz : length-3 tuple or None
        Caller's override.
    nz_default : length-3 tuple
        Build-time normalised n(z) per leg, or None.
    z_arrs : length-3 tuple
        Per-leg z-grids, or None.
    N_cosmo : int
    jdtype : JAX dtype

    Returns
    -------
    nz : length-3 tuple
        Each entry (N_cosmo, N_z_i) or None.
    """
    if nz is None:
        nz = (None, None, None)
    if len(nz) != 3:
        raise ValueError(
            f"nz must be a length-3 tuple, one entry per leg (got {len(nz)})."
        )

    out = []
    for i in range(3):
        if nz_default[i] is None:
            if nz[i] is not None:
                raise ValueError(
                    f"nz[{i}] must be None: leg {i} has no source distribution "
                    "(its tracer takes no dndz)."
                )
            out.append(None)
            continue
        v = jnp.asarray(nz_default[i] if nz[i] is None else nz[i], dtype=jdtype)
        n_z = len(z_arrs[i])
        if v.shape[-1] != n_z:
            raise ValueError(
                f"nz[{i}] has trailing dimension {v.shape[-1]}, but leg {i}'s "
                f"z-grid has {n_z} points. The z-grid is fixed at build time; "
                "pass n(z) sampled on that same grid."
            )
        if v.ndim == 1:
            v = jnp.broadcast_to(v, (N_cosmo, n_z))
        elif v.ndim != 2 or v.shape[0] != N_cosmo:
            raise ValueError(
                f"nz[{i}] must have shape ({n_z},) or ({N_cosmo}, {n_z}) "
                f"(got {v.shape})."
            )
        if nz[i] is not None:
            _check_nz_values(nz[i], z_arrs[i], f"nz[{i}]")
        out.append(v)
    return tuple(out)


def _resolve_bias(bias, bias_default, N_cosmo, jdtype):
    """Per-leg clustering bias as (N_cosmo,), or None where a leg has none.

    ``bias`` is the caller's override (a length-3 tuple, entries None to keep
    the build-time default); a scalar is shared across the cosmology batch.

    Parameters
    ----------
    bias : length-3 tuple or None
        Caller's override.
    bias_default : length-3 tuple
        Build-time bias per leg, or None.
    N_cosmo : int
    jdtype : JAX dtype

    Returns
    -------
    bias : length-3 tuple
        Each entry (N_cosmo,) or None.
    """
    if bias is None:
        bias = (None, None, None)
    if len(bias) != 3:
        raise ValueError(
            f"bias must be a length-3 tuple, one entry per leg (got {len(bias)})."
        )

    out = []
    for i in range(3):
        if bias_default[i] is None:
            if bias[i] is not None:
                raise ValueError(
                    f"bias[{i}] must be None: leg {i} is not a 'counts' leg."
                )
            out.append(None)
            continue
        v = jnp.asarray(bias_default[i] if bias[i] is None else bias[i], dtype=jdtype)
        if v.ndim == 0:
            v = jnp.broadcast_to(v, (N_cosmo,))
        elif v.shape != (N_cosmo,):
            raise ValueError(
                f"bias[{i}] must be a scalar or have shape ({N_cosmo},) "
                f"(got {v.shape})."
            )
        out.append(v)
    return tuple(out)


def _resolve_ia(ia, has_ia_leg, N_cosmo, jdtype):
    """The alignment parameters as ``(A_IA, eta)``, each ``(N_cosmo,)``, or None.

    Required exactly when some leg is an 'ia' leg; passing it otherwise is an
    error rather than being ignored, as for ``bias``.

    Parameters
    ----------
    ia : IA, dict, tuple, number or None
        Anything :func:`as_ia` accepts.
    has_ia_leg : bool
    N_cosmo : int
    jdtype : JAX dtype

    Returns
    -------
    ia : (A_IA, eta) tuple of (N_cosmo,) arrays, or None
    """
    if not has_ia_leg:
        if ia is not None:
            raise ValueError(
                "ia must be None: no leg is an 'ia' tracer, so it would be ignored."
            )
        return None
    if ia is None:
        raise ValueError(
            "ia is required because a leg is an 'ia' tracer; "
            "pass ia=IA(A_IA=..., eta=...)."
        )
    out = []
    for name, v in zip(IA._fields, as_ia(ia)):
        v = jnp.asarray(0.0 if v is None else v, dtype=jdtype)
        if v.ndim == 0:
            v = jnp.broadcast_to(v, (N_cosmo,))
        elif v.shape != (N_cosmo,):
            raise ValueError(
                f"ia.{name} must be a scalar or have shape ({N_cosmo},) "
                f"(got {v.shape})."
            )
        out.append(v)
    return tuple(out)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def make_projected_bihalofit(
    ell1,
    ell2,
    ell3,
    dndz,
    tracers,
    bias=(None, None, None),
    z_source=(1089.0, 1089.0, 1089.0),
    pk_source="syren",
    N_chi=160,
    chi_min=_CHI_MIN_DEFAULT,
    chi_max=_CHI_MAX_DEFAULT,
    N_R=15,
    tol=None,
    R_min=1e-6,
    R_max=40.0,
    dtype=np.float32,
    validate=False,
    term="full",
    baryonic_boost=False,
    k_lin=None,
    N_z_bg=2000,
    limber_shift=0.5,
    z_pivot=_IA_Z_PIVOT_DEFAULT,
):
    r"""Build a Limber-projected bispectrum callable for a fixed geometry, with one tracer per leg.

    Set-up and JIT compilation happen once; call the result at many cosmologies.

    Parameters
    ----------
    ell1, ell2, ell3 : array_like (N_ltri,)
        Multipoles of the triangle sides.
    dndz : tuple of 3 ``(z_arr, nz_arr)`` or None
        Per-leg n(z), sampled at points (pass bin centres for binned data) on a grid starting at
        z=0. Required for ``'lensing'``, ``'counts'`` and ``'ia'`` legs, ``None`` for
        ``'cmb_lensing'``. The lensing efficiency is integrated on this grid as given; a warning
        fires below 50 points.
    tracers : tuple of 3 str
        ``'lensing'``, ``'counts'`` (clustering, no magnification), ``'cmb_lensing'`` (one source
        plane at ``z_source``) or ``'ia'`` (NLA intrinsic alignments). The shear terms GGG, GGI,
        GII and III are combinations of ``'lensing'`` and ``'ia'`` legs, summed by the caller.
    bias : tuple of 3 float or None, optional
        Clustering bias, required exactly for ``'counts'`` legs; overridable per call.
    z_source : tuple of 3 float, optional
        Source redshift of ``'cmb_lensing'`` legs (default 1089). The default ``chi_max`` cuts
        part of a CMB-lensing kernel, which matters only when every leg is CMB lensing.
    pk_source : {'syren', 'baccoemu', 'eh', 'eh_99'}, optional
        Linear P(k) backend (default 'syren'), as in :func:`~bihalofit_jax.make_bihalofit`.
    N_chi : int, optional
        Gauss-Legendre nodes in :math:`\chi` (default 160): mean error 0.001% for lensing and IA
        legs, up to 0.05% with three ``'counts'`` legs. Narrow counts n(z) need
        :math:`N_\chi\gtrsim 8000\,\mathrm{Mpc}/h` over their radial width; a warning says when.
    chi_min, chi_max : float, optional
        :math:`\chi` bounds in Mpc/h.
    N_R, tol, R_min, R_max, dtype, validate, term, baryonic_boost, k_lin
        As in :func:`~bihalofit_jax.make_bihalofit`; ``tol`` takes the worst :math:`\chi` node.
    N_z_bg : int, optional
        Points of the uniform part (:math:`z\le10`) of the background grid that maps
        :math:`\chi` to :math:`z` (default 2000); 500 log-spaced points continue to ``z_source``.
    limber_shift : float, optional
        :math:`k_i = (\ell_i + \mathrm{limber\_shift})/\chi`. Default 0.5 (LoVerde & Afshordi
        2008), with error :math:`O(1/\ell^2)`; 0 is the plain Limber convention of e.g.
        ``threepoint``, for comparisons with it.
    z_pivot : float, optional
        Pivot redshift of the ``'ia'`` amplitude (default 0.62).

    Returns
    -------
    projected_bihalofit : callable
        ``projected_bihalofit(cosmo, nz=None, bias=None, ia=None)`` -> (N_cosmo, N_ltri), for a
        :class:`~bihalofit_jax.Cosmology`. Pure and differentiable unless ``tol`` or ``validate``
        is set, which syncs to the host to check.

        ``nz`` and ``bias`` override the build-time values per leg (``None`` keeps a leg's), and
        are differentiable. An ``nz`` entry is ``(N_z,)`` or ``(N_cosmo, N_z)`` on that leg's
        build-time grid, normalised in the trace; it must be non-negative with a positive
        integral. A ``bias`` entry is a scalar or ``(N_cosmo,)``. ``ia`` is an
        :class:`~bihalofit_jax.IA` ``(A_IA, eta)``, required exactly when a leg is ``'ia'``.
        The z grids are fixed at build time.

    Notes
    -----
    For ``'lensing'``,

    .. math::
        W(\chi) = \frac{3}{2}\, \Omega_m \left(\frac{H_0}{c}\right)^2
            \frac{\chi}{a(\chi)}\, g(\chi), \qquad
        g(\chi) = \int_\chi^{\chi_H} n(z')\,
            \frac{\chi(z') - \chi}{\chi(z')}\, dz'

    For ``'counts'``, :math:`W(\chi) = b\, n_g(z(\chi))\, H(z(\chi))/c`; ``'cmb_lensing'`` is the
    lensing form with one source plane. For ``'ia'``, :math:`W(\chi) = f_\mathrm{IA}(z)\, n(z)\,
    H(z)/c` with :math:`f_\mathrm{IA} = -A_\mathrm{IA}\, 0.0134\, \Omega_m/D(z)\,
    [(1+z)/(1+z_\mathrm{pivot})]^\eta` and :math:`D(0)=1`, so the mixed bispectra are
    :math:`B_{\delta\delta I}=f_\mathrm{IA} B_{\delta\delta\delta}` and so on.
    """
    check_pk_options(pk_source, term, k_lin)
    validate_tracer_config(dndz, tracers, bias, z_source)
    if not (np.isfinite(z_pivot) and z_pivot > -1.0):
        raise ValueError(
            f"z_pivot must be finite and greater than -1 (got {z_pivot!r})."
        )
    has_ia = "ia" in tracers

    N_R = _resolve_N_R(N_R, tol)

    jdtype = resolve_dtype(dtype)

    ell1 = np.asarray(ell1, dtype=float)
    ell2 = np.asarray(ell2, dtype=float)
    ell3 = np.asarray(ell3, dtype=float)

    # Validate and normalize each leg's own dndz: chi(z) assumes z_arr[0]==0,
    # and B's amplitude depends on each n(z)'s absolute normalization.
    z_arrs = [None, None, None]
    nz_arrs = [None, None, None]
    for i in range(3):
        if dndz[i] is None:
            continue
        z_arr = np.ascontiguousarray(dndz[i][0], dtype=float)
        nz_arr = np.ascontiguousarray(dndz[i][1], dtype=float)
        if z_arr.ndim != 1 or len(z_arr) < 2 or not np.all(np.diff(z_arr) > 0):
            raise ValueError(
                f"dndz[{i}][0] must be a 1-D strictly increasing redshift grid."
            )
        if z_arr[0] != 0.0:
            raise ValueError(
                f"dndz[{i}][0] must start at z=0 (got z[0]={z_arr[0]}); "
                "χ(z) assumes χ(0)=0."
            )
        _check_nz_values(nz_arr, z_arr, f"dndz[{i}][1]")
        nz_norm = _np_trapezoid(nz_arr, z_arr)
        z_arrs[i] = z_arr
        nz_arrs[i] = nz_arr / nz_norm
    dndz_norm = tuple(
        (z_arrs[i], nz_arrs[i]) if z_arrs[i] is not None else None for i in range(3)
    )

    # Captured under its own name: `bias` is shadowed inside the returned
    # callable, where it is the caller's per-call override.
    bias_static = tuple(bias)

    chi_nodes, gl_weights = gauss_legendre(chi_min, chi_max, N_chi)

    z_bg_max = _background_zmax(chi_max, tracers, z_source)
    z_bg = _background_zgrid(z_bg_max, N_z_bg)

    # ---- setup diagnostics (first build for this geometry only) ----
    # Approximate, warning-only checks; see _diagnostics.py.
    warn_if_geometry_inadequate(
        _DiagnosticInputs(
            ell1=ell1,
            ell2=ell2,
            ell3=ell3,
            chi_nodes=chi_nodes,
            gl_weights=gl_weights,
            dndz=dndz_norm,
            tracers=tracers,
            bias=bias,
            z_source=z_source,
            chi_max=chi_max,
            pk_source=pk_source,
            k_lin=k_lin,
            limber_shift=limber_shift,
        )
    )

    batched = jax.jit(
        jax.vmap(
            _build_limber_kernel(
                ell1,
                ell2,
                ell3,
                dndz_norm,
                tracers,
                z_source,
                z_bg,
                chi_nodes,
                gl_weights,
                pk_source,
                N_R,
                R_min,
                R_max,
                jdtype,
                term=term,
                baryonic_boost=baryonic_boost,
                k_lin=k_lin,
                limber_shift=limber_shift,
                z_pivot=z_pivot,
            )
        )
    )

    def projected_bihalofit(cosmo, nz=None, bias=None, ia=None):
        """Cosmology (plus optional per-call nz/bias/ia overrides) -> B, (N_cosmo, N_ltri).

        See the Returns section of :func:`make_projected_bihalofit`.
        """
        cosmo = as_cosmology(cosmo)

        # Bounds check before any H2D transfer; the expfactor range was checked at setup.
        if pk_source == "baccoemu":
            _validate_baccoemu_bounds(cosmo._asdict())

        p = cosmo.as_arrays(jdtype)
        args = (
            p.omega_cold,
            p.omega_baryon,
            p.sigma8_cold,
            p.ns,
            p.neutrino_mass,
            p.w0,
            p.wa,
            p.hubble,
        )
        N_cosmo = p.omega_cold.shape[0]
        args = args + (
            _resolve_nz(nz, nz_arrs, z_arrs, N_cosmo, jdtype),
            _resolve_bias(bias, bias_static, N_cosmo, jdtype),
            _resolve_ia(ia, has_ia, N_cosmo, jdtype),
        )

        # As in make_bihalofit's callable: the tol check runs whenever tol is
        # set, and validate picks raise-vs-NaN.
        if tol is None and not validate:
            B_out = batched(*args)[0]
        else:
            B_out, resid_out = batched(*args)
            B_out = check_solver(
                B_out,
                resid_out,
                tol=tol,
                validate=validate,
                N_R=N_R,
                jdtype=jdtype,
                over="cosmology/chi-node",
            )

        # Always (N_cosmo, N_ltri), even for scalar cosmology fields.
        return B_out

    return projected_bihalofit
