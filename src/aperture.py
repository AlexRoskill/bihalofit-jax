r"""Third-order aperture statistics :math:`\langle X_1 X_2 X_3\rangle` from the Limber-projected
bispectrum, with one tracer per leg (Heydenreich et al. 2023, Burger et al. 2024):

.. math::
    \langle X_1 X_2 X_3 \rangle(\theta_1, \theta_2, \theta_3) =
        \frac{2}{8\pi^3} \int_0^\infty d\ell_1 \int_0^\infty d\ell_2
        \int_0^\pi d\varphi\; \ell_1 \ell_2\,
        B(\ell_1, \ell_2, \ell_3)\,
        \hat u(\ell_1\theta_1)\, \hat u(\ell_2\theta_2)\, \hat u(\ell_3\theta_3)

with :math:`\ell_3^2 = \ell_1^2 + \ell_2^2 + 2\ell_1\ell_2\cos\varphi` and the Crittenden (2002)
filter :math:`\hat u(\eta) = \tfrac12 \eta^2 e^{-\eta^2/2}`. ``'lensing'`` on every leg gives
:math:`\langle M_\mathrm{ap}^3\rangle`. Every tracer gets the same filter, the usual generalisation;
check it against your survey's convention before using mixed tracers.

:math:`B` is evaluated once on a fixed Gauss-Legendre :math:`(\ell_1, \ell_2, \varphi)` grid, so any
number of triplets is cheap. Angles are in radians.
"""

import warnings

import numpy as np
import jax
import jax.numpy as jnp

from ._config import resolve_dtype
from ._utils import gauss_legendre, gauss_legendre_log
from .projected_bispectrum import (
    make_projected_bihalofit,
    _CHI_MIN_DEFAULT,
    _CHI_MAX_DEFAULT,
)

ARCMIN_TO_RAD = np.pi / 180.0 / 60.0

_MAP3_PREFACTOR = 2.0 / (8.0 * np.pi**3)

# u_hat(eta) = 0.5 eta^2 exp(-eta^2/2) is negligible beyond eta ~ 10, so the
# ell integral only needs to reach ell ~ 10 / theta. Same theta*l <= 10
# convention Heydenreich's `threepoint` uses to set its own per-triplet lMax
# (threepoint/classed_version/apertureStatistics.cpp::MapMapMap).
_ELL_THETA_PRODUCT = 10.0

# ell_max when neither it nor theta_min is given. Too low for apertures below ~1.7 arcmin.
_ELL_MAX_LEGACY_DEFAULT = 2.0e4


def u_hat(eta):
    r"""Crittenden (2002) aperture filter in Fourier space,
    :math:`\hat u(\eta) = \tfrac12\eta^2 e^{-\eta^2/2}`.

    Parameters
    ----------
    eta : array
        :math:`\ell\theta`, with :math:`\theta` in radians.

    Returns
    -------
    u : shape of eta
    """
    half_eta2 = 0.5 * eta**2
    return half_eta2 * jnp.exp(-half_eta2)


def ratio_nodes(N_rho, rho_max, order=2, tightest=3e-3):
    r"""Nodes/weights for :math:`\int f(\rho)\,d\rho` over
    :math:`\rho = \ell_2/\ell_1`, geometrically dense at :math:`\rho = 1`:
    ``N_rho`` panels (``N_rho // 2`` on each side of 1) of ``order`` nodes each.

    Panel edges are placed at :math:`|\ln\rho| =` ``tightest`` ... ``ln(rho_max)``,
    geometrically spaced and mirrored about :math:`\rho=1`, with an order-
    ``order`` GL rule per panel. So resolution is concentrated exactly where
    :math:`\ell_1 \approx \ell_2`, and ``tightest`` sets the narrowest ratio
    window resolved (default 0.3%).

    Parameters
    ----------
    N_rho : int
        Number of panels (rounded down to even, at least 2).
    rho_max : float
        Largest ratio; nodes span [1/rho_max, rho_max].
    order : int, optional
        GL nodes per panel.
    tightest : float, optional
        Innermost panel edge in :math:`|\ln\rho|`.

    Returns ``(rho, weights)`` with the :math:`d\rho = \rho\,d\ln\rho`
    Jacobian already folded in, so ``sum(weights * f(rho))`` approximates
    :math:`\int f\,d\rho` directly.
    """
    half = max(1, N_rho // 2)
    u = np.geomspace(tightest, np.log(rho_max), half)
    edges = np.concatenate([-u[::-1], [0.0], u])
    xi, wi = np.polynomial.legendre.leggauss(order)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (hi - lo) * xi + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * wi)
    ln_rho = np.concatenate(xs)
    rho = np.exp(ln_rho)
    return rho, np.concatenate(ws) * rho


def phi_nodes_bunched(N_per, n_panels, closest=2e-3):
    r"""Composite GL nodes on :math:`[0,\pi]` with panels geometrically dense
    toward :math:`\varphi=\pi`.

    On the diagonal (:math:`\ell_1=\ell_2`) the closing side is
    :math:`\ell_3 = 2\ell_1\cos(\varphi/2)`, so the third filter
    :math:`\hat u(\ell_3\theta_3)` peaks at
    :math:`\varphi = \pi - \sqrt{2}/(\ell_1\theta_3)`: a narrow bump against the
    endpoint, which a global rule resolves only with many nodes. Panels bunched at
    :math:`\pi` reach the same accuracy with about half of them. Only useful with
    ``grid='diagonal'``.

    Parameters
    ----------
    N_per : int
        GL nodes per panel.
    n_panels : int
        Number of panels.
    closest : float, optional
        Width of the panel nearest pi, as a fraction of pi.

    Returns
    -------
    phi, w : (N_per * n_panels,) ndarray
        Nodes and weights (at most; coincident edges are merged).
    """
    frac = np.geomspace(closest, 1.0, n_panels)
    edges = np.unique(np.concatenate([[0.0], np.pi * (1.0 - frac[::-1]), [np.pi]]))
    xi, wi = np.polynomial.legendre.leggauss(N_per)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (hi - lo) * xi + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * wi)
    return np.concatenate(xs), np.concatenate(ws)


def _build_triangle_grid_diagonal(
    ell_min,
    ell_max,
    N_ell,
    N_rho,
    N_phi,
    rho_order=2,
    rho_tightest=3e-3,
    phi_bunch=None,
):
    r"""Triangle grid in :math:`(\ell_1, \rho, \varphi)` with
    :math:`\ell_2 = \ell_1\rho` -- the "diagonal" scheme.

    When one aperture is much larger than the other two, its filter needs small
    :math:`\ell_3`; but :math:`\ell_3 \ge |\ell_1-\ell_2|`, so the whole contribution
    sits in a thin band around :math:`\ell_1 \approx \ell_2`, whose width in
    :math:`\ln\ell` shrinks like :math:`1/\ell`. Sampling :math:`\ell_2` independently
    cannot resolve that band; sampling the ratio with nodes dense at :math:`\rho=1`
    resolves it by construction.

    ``ell_2`` outside ``[ell_min, ell_max]`` is dropped, so the integration domain matches
    the ``(ell, ell, phi)`` grid's.

    Parameters
    ----------
    ell_min, ell_max : float
        Multipole range of ell1 and ell2.
    N_ell : int
        Log-GL nodes in ell1.
    N_rho : int
        Ratio panels; see :func:`ratio_nodes`.
    N_phi : int
        GL nodes in phi (ignored if ``phi_bunch`` is given).
    rho_order, rho_tightest : optional
        Passed to :func:`ratio_nodes` as ``order`` and ``tightest``.
    phi_bunch : (N_per, n_panels) tuple, optional
        Use :func:`phi_nodes_bunched` instead of a global phi rule.

    Returns
    -------
    ell1, ell2, ell3, weight : flattened arrays, as from :func:`_build_triangle_grid`.
        ``weight`` carries the :math:`d\ell_1 d\ell_2 = \ell_1\,d\ell_1\,d\rho` Jacobian
        but not the ``ell1*ell2`` factor of the integrand, which make_map3 applies.
    """
    ell_nodes, w_ell = gauss_legendre_log(ell_min, ell_max, N_ell)
    rho, w_rho = ratio_nodes(N_rho, ell_max / ell_min, rho_order, rho_tightest)
    if phi_bunch is None:
        phi_nodes, w_phi = gauss_legendre(0.0, np.pi, N_phi)
    else:
        phi_nodes, w_phi = phi_nodes_bunched(*phi_bunch)

    E1, RHO, PHI = np.meshgrid(ell_nodes, rho, phi_nodes, indexing="ij")
    W1, WR, WPHI = np.meshgrid(w_ell, w_rho, w_phi, indexing="ij")
    E1, RHO, PHI = E1.ravel(), RHO.ravel(), PHI.ravel()
    E2 = E1 * RHO
    weight = (W1 * WR * WPHI).ravel() * E1  # dl1 dl2 = l1 dl1 drho

    keep = (E2 >= ell_min) & (E2 <= ell_max)
    E1, E2, PHI, weight = E1[keep], E2[keep], PHI[keep], weight[keep]
    ell3 = np.sqrt(np.maximum(E1**2 + E2**2 + 2.0 * E1 * E2 * np.cos(PHI), 0.0))
    return E1, E2, ell3, weight


def _build_triangle_grid(ell_min, ell_max, N_ell, N_phi):
    """Flattened GL quadrature grid of (ell1, ell2, phi) triangles for the
    Map3 angular double integral, plus the combined ready-to-sum weight.

    Every (ell1, ell2, phi) combination is a geometrically valid Fourier
    triangle (ell3 from the law of cosines is always real and positive), so
    no masking is needed -- unlike matter-space k-triangles.

    ``ell1``/``ell2`` each use a global log-GL rule
    (:func:`gauss_legendre_log`) of order ``N_ell``.

    Parameters
    ----------
    ell_min, ell_max : float
        Multipole range of ell1 and ell2.
    N_ell : int
        Log-GL nodes per ell axis.
    N_phi : int
        GL nodes in phi on [0, pi].

    Returns
    -------
    ell1, ell2, ell3, weight : (N_ell**2*N_phi,) flattened numpy
        arrays. weight already includes the log-spacing Jacobians for ell1,
        ell2 and the GL weight for phi; it does NOT include the ell1*ell2
        factor from the physical Map3 integrand -- that is applied
        explicitly at reduction time (see make_map3), matching the integral
        as written above.
    """
    ell_nodes, w_ell = gauss_legendre_log(ell_min, ell_max, N_ell)
    phi_nodes, w_phi = gauss_legendre(0.0, np.pi, N_phi)

    E1, E2, PHI = np.meshgrid(ell_nodes, ell_nodes, phi_nodes, indexing="ij")
    W1, W2, WPHI = np.meshgrid(w_ell, w_ell, w_phi, indexing="ij")

    ell3 = np.sqrt(E1**2 + E2**2 + 2.0 * E1 * E2 * np.cos(PHI))
    weight = W1 * W2 * WPHI

    return E1.ravel(), E2.ravel(), ell3.ravel(), weight.ravel()


def make_map3(
    dndz,
    tracers,
    bias=(None, None, None),
    z_source=(1089.0, 1089.0, 1089.0),
    ell_min=1.0,
    ell_max=None,
    theta_min=None,
    N_ell=96,
    N_phi=64,
    grid="diagonal",
    N_rho=48,
    phi_bunch=None,
    pk_source="syren",
    N_chi=160,
    chi_min=_CHI_MIN_DEFAULT,
    chi_max=_CHI_MAX_DEFAULT,
    N_R=15,
    tol=None,
    R_min=1e-6,
    R_max=40.0,
    dtype=np.float32,
    term="full",
    baryonic_boost=False,
    k_lin=None,
    N_z_bg=2000,
    limber_shift=0.5,
    z_pivot=0.62,
):
    r"""Build a :math:`\langle X_1 X_2 X_3 \rangle` callable for fixed tracers and quadrature.

    The grid, the projected-bispectrum set-up and JIT compilation happen once; call the result
    at many aperture triplets and cosmologies.

    Parameters
    ----------
    dndz, tracers, bias, z_source
        As in :func:`~bihalofit_jax.make_projected_bihalofit`. ``('lensing',)*3`` with one n(z)
        gives :math:`\langle M_\mathrm{ap}^3\rangle`.
    ell_min, ell_max : float, optional
        Bounds of the :math:`\ell_1, \ell_2` quadrature. ``ell_max=None`` takes
        :math:`10/\theta_\mathrm{min}`, or ``2e4`` without ``theta_min``, which truncates the
        integral for apertures below ~1.7'.
    theta_min : float, optional
        Smallest aperture radius in radians that will be evaluated, used to size ``ell_max``.
    grid : {'diagonal', 'phi'}, optional
        ``'diagonal'`` (default) samples the ratio :math:`\rho=\ell_2/\ell_1` densely at 1 and is
        far more accurate; ``'phi'`` samples :math:`\ell_2` directly, as an independent check.
    N_ell, N_rho, N_phi : int, optional
        Gauss-Legendre nodes in :math:`\ln\ell_1`, panels (of two nodes) in :math:`\rho`, and
        nodes in :math:`\varphi`. The defaults 96, 48, 64 reach 1.6e-4 at worst on 1' to 64'
        apertures; larger apertures need more :math:`\varphi` nodes. Pass the largest aperture
        last: the grid is accurate in that order.
    phi_bunch : (int, int) or None, optional
        ``(N_per, n_panels)`` for :func:`phi_nodes_bunched`, panels dense at :math:`\varphi=\pi`
        (diagonal grid only). ``None`` is a global rule.
    pk_source, N_chi, chi_min, chi_max, N_R, tol, R_min, R_max, dtype, term, baryonic_boost, k_lin, N_z_bg, limber_shift, z_pivot
        As in :func:`~bihalofit_jax.make_projected_bihalofit`. The default P(k) grid ends at
        50 h/Mpc, so apertures below about :math:`3'` extrapolate it and the build warns; for cosmic
        shear this changes :math:`\langle M_\mathrm{ap}^3\rangle` by about :math:`10^{-6}` from
        :math:`2'` to :math:`64'`, and a wider ``k_lin`` removes it.

    Returns
    -------
    map3 : callable
        ``map3(theta1, theta2, theta3, cosmo, ia=None)`` -> (N_cosmo, N_theta), for
        broadcastable radii in radians and a :class:`~bihalofit_jax.Cosmology`; ``ia`` is an
        :class:`~bihalofit_jax.IA`, required exactly when a leg is ``'ia'``. Differentiable.
    """
    if ell_max is None:
        ell_max = (
            _ELL_THETA_PRODUCT / float(theta_min)
            if theta_min is not None
            else _ELL_MAX_LEGACY_DEFAULT
        )
    elif theta_min is not None:
        raise ValueError(
            "Pass either ell_max or theta_min, not both -- theta_min exists "
            "only to derive ell_max, so giving both is ambiguous."
        )

    if grid == "phi":
        ell1, ell2, ell3, gl_weight = _build_triangle_grid(
            ell_min, ell_max, N_ell, N_phi
        )
    elif grid == "diagonal":
        ell1, ell2, ell3, gl_weight = _build_triangle_grid_diagonal(
            ell_min, ell_max, N_ell, N_rho, N_phi, phi_bunch=phi_bunch
        )
    else:
        raise ValueError(f"grid must be 'phi' or 'diagonal' (got {grid!r}).")

    projected_bihalofit = make_projected_bihalofit(
        ell1,
        ell2,
        ell3,
        dndz,
        tracers,
        bias=bias,
        z_source=z_source,
        pk_source=pk_source,
        N_chi=N_chi,
        chi_min=chi_min,
        chi_max=chi_max,
        N_R=N_R,
        tol=tol,
        R_min=R_min,
        R_max=R_max,
        dtype=dtype,
        term=term,
        baryonic_boost=baryonic_boost,
        k_lin=k_lin,
        N_z_bg=N_z_bg,
        limber_shift=limber_shift,
        z_pivot=z_pivot,
    )

    jdtype = resolve_dtype(dtype)
    ell1_j = jnp.asarray(ell1, dtype=jdtype)
    ell2_j = jnp.asarray(ell2, dtype=jdtype)
    ell3_j = jnp.asarray(ell3, dtype=jdtype)
    w_j = jnp.asarray(gl_weight, dtype=jdtype)

    def _map3_one_triplet(theta1, theta2, theta3, B_row):
        """<Map^3> for scalar radii (radians) from one cosmology's B on the grid, (N_grid,) -> scalar."""
        integrand = (
            ell1_j
            * ell2_j
            * B_row
            * u_hat(ell1_j * theta1)
            * u_hat(ell2_j * theta2)
            * u_hat(ell3_j * theta3)
        )
        return _MAP3_PREFACTOR * jnp.sum(w_j * integrand)

    _map3_all_triplets = jax.vmap(_map3_one_triplet, in_axes=(0, 0, 0, None))

    def _map3_all_cosmologies(th1, th2, th3, B):
        """Radii (N_theta,) each and B (N_cosmo, N_grid) -> <Map^3>, (N_cosmo, N_theta)."""
        return jax.vmap(lambda B_row: _map3_all_triplets(th1, th2, th3, B_row))(B)

    # Jitted once here, and deliberately not fused with projected_bihalofit into one jit:
    # that measured 3-26% slower, as two dispatches pipeline in the GPU's queue.
    _map3_jit = jax.jit(_map3_all_cosmologies)

    def map3(theta1, theta2, theta3, cosmo, ia=None):
        """Broadcastable radii in radians, a Cosmology and optional IA -> <Map^3>, (N_cosmo, N_theta).

        See the Returns section of :func:`make_map3`.
        """
        # Warn if the smallest aperture needs more ell than this grid spans (see ell_max
        # above). Only possible with concrete thetas.
        try:
            _th_min = float(
                np.min(
                    [
                        np.min(np.asarray(t, dtype=float))
                        for t in (theta1, theta2, theta3)
                    ]
                )
            )
        except (TypeError, ValueError):
            _th_min = None
        if _th_min is not None and _th_min > 0.0:
            _need = _ELL_THETA_PRODUCT / _th_min
            if _need > ell_max * (1.0 + 1e-9):
                warnings.warn(
                    f"theta_min={_th_min:.3e} rad needs ell_max >= {_need:.0f} "
                    f"(theta*l <= {_ELL_THETA_PRODUCT:g}) but this grid stops at "
                    f"{ell_max:.0f}; the ell integral is truncated and "
                    f"<Map^3> will be biased low for shapes weighted at high "
                    f"ell. Rebuild with theta_min={_th_min:.3e} (or an explicit "
                    f"ell_max).",
                    RuntimeWarning,
                    stacklevel=2,
                )

        B = projected_bihalofit(cosmo, ia=ia)  # (N_cosmo, N_grid)

        th1, th2, th3 = jnp.broadcast_arrays(
            jnp.atleast_1d(jnp.asarray(theta1, dtype=jdtype)),
            jnp.atleast_1d(jnp.asarray(theta2, dtype=jdtype)),
            jnp.atleast_1d(jnp.asarray(theta3, dtype=jdtype)),
        )

        # Consistent rank: always (N_cosmo, N_theta).
        return _map3_jit(th1, th2, th3, B)

    return map3
