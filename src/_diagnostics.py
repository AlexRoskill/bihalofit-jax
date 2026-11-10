"""Set-up warnings for the projected bispectrum: are the chi bounds and the P(k) grid adequate for
the requested geometry? They use one fiducial cosmology and a BBKS P(k), which is enough to place a
warning, and never touch the differentiable computation.
"""

import warnings
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from .background import _comoving_distance, lensing_geometry
from ._pk_backends import _STANDARD_K_GRID, _get_baccoemu

# np.trapezoid added in NumPy 2.0; np.trapz removed in 2.0.
try:
    _np_trapezoid = np.trapezoid
except AttributeError:
    _np_trapezoid = np.trapz


class _DiagnosticInputs(NamedTuple):
    """The setup values the diagnostics need, bundled so they travel as one.

    Built once by make_projected_bihalofit after its own validation and
    normalisation, so dndz here is already normalised to unit integral.
    """

    ell1: np.ndarray
    ell2: np.ndarray
    ell3: np.ndarray
    chi_nodes: np.ndarray
    gl_weights: np.ndarray
    dndz: tuple
    tracers: tuple
    bias: tuple
    z_source: tuple
    chi_max: float
    pk_source: str
    k_lin: np.ndarray = None  # None = the built-in _STANDARD_K_GRID
    limber_shift: float = 0.5


def _bbks_pk(k, Omega_m=0.3, h=0.7, ns=0.96):
    r"""Crude BBKS linear P(k) [arbitrary units], valid at all k (numpy).

    Used only as a shape proxy for :math:`B_\delta \propto P(k)^2` when judging
    whether the :math:`\chi`-nodes that leave the P_L grid carry real weight.
    The steep high-k falloff is what matters; the overall normalization
    cancels in the relative weighting.

    Parameters
    ----------
    k : array-like
        Wavenumbers in h/Mpc.
    Omega_m, h, ns : float, optional
        Fiducial cosmology.

    Returns
    -------
    pk : ndarray, shape of k
    """
    q = np.asarray(k, dtype=float) / (Omega_m * h)  # k in h/Mpc
    q = np.where(q > 0, q, 1e-30)
    T = (
        np.log(1.0 + 2.34 * q)
        / (2.34 * q)
        * (1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4)
        ** -0.25
    )
    return k**ns * T**2


def _fiducial_leg_weight(
    tracer, chi_nodes, chi_z_probe, z_probe, dndz_i, bias_i, z_source_i, Omega_m
):
    """Fiducial (numpy) ``|W_i(chi)|`` shape for one leg, for the node-weight
    diagnostic only (_fiducial_node_weights): not the differentiable path.

    Parameters
    ----------
    tracer : {'lensing', 'counts', 'ia', 'cmb_lensing'}
    chi_nodes : (N_chi,) ndarray
        Comoving distances in Mpc/h.
    chi_z_probe, z_probe : (N_probe,) ndarray
        Fiducial chi(z) lookup table.
    dndz_i : (z_arr, nz_arr) tuple
        This leg's n(z); unused for 'cmb_lensing'.
    bias_i : float
        Linear bias, used for 'counts'.
    z_source_i : float
        Source redshift, used for 'cmb_lensing'.
    Omega_m : float

    Returns
    -------
    W : (N_chi,) ndarray
        Kernel weight per node (arbitrary normalisation).
    """
    if tracer == "lensing":
        z_arr, nz_arr = dndz_i
        chi_z = np.asarray(_comoving_distance(jnp.asarray(z_arr), Omega_m, -1.0, 0.0))
        z_n = np.interp(chi_nodes, chi_z, z_arr)
        a_n = 1.0 / (1.0 + z_n)
        g_n = np.asarray(
            lensing_geometry(
                jnp.asarray(chi_nodes),
                jnp.asarray(chi_z),
                jnp.asarray(nz_arr),
                jnp.asarray(z_arr),
            )
        )
        return chi_nodes / a_n * g_n
    if tracer in ("counts", "ia"):  # 'ia': the slowly varying f_IA(z) is taken as 1
        z_arr, nz_arr = dndz_i
        z_n = np.interp(chi_nodes, chi_z_probe, z_probe)
        E_n = np.sqrt(Omega_m * (1.0 + z_n) ** 3 + (1.0 - Omega_m))
        ng_n = np.interp(z_n, z_arr, nz_arr, left=0.0, right=0.0)
        return (1.0 if tracer == "ia" else bias_i) * ng_n * E_n
    # 'cmb_lensing'
    z_n = np.interp(chi_nodes, chi_z_probe, z_probe)
    a_n = 1.0 / (1.0 + z_n)
    chi_star = float(np.interp(z_source_i, z_probe, chi_z_probe))
    return chi_nodes / a_n * np.clip((chi_star - chi_nodes) / chi_star, 0.0, None)


def _fiducial_node_weights(
    chi_nodes, gl_weights, tracers, dndz, bias, z_source, ell_ref, Omega_m=0.3
):
    r"""Relative ``|Limber integrand|`` per :math:`\chi`-node for a fiducial cosmology.

    The integrand is :math:`\mathrm{gl\_w}/\chi^4 \cdot W_1 W_2 W_3 \cdot B_\delta(\ell/\chi)`.
    The geometric factor alone peaks at small :math:`\chi`, but
    :math:`B_\delta` falls steeply there since :math:`k=\ell/\chi` is large,
    so we fold in a BBKS :math:`P(k_\mathrm{ref})^2` proxy to reflect where
    :math:`B` actually gets its support, the basis for gating an
    extrapolation warning.

    Parameters
    ----------
    chi_nodes, gl_weights : (N_chi,) ndarray
        Limber quadrature nodes (Mpc/h) and weights.
    tracers, dndz, bias, z_source : length-3 tuples
        Per-leg tracer settings, as in ``_DiagnosticInputs``.
    ell_ref : float
        Representative multipole for the P(k)^2 proxy.
    Omega_m : float, optional

    Returns
    -------
    w : (N_chi,) ndarray
        Relative weight per node, normalised to sum 1 (unless it is all zero).
    """
    z_probe_hi = max(20.0, 1.05 * max((float(zs) for zs in z_source), default=0.0))
    z_probe = np.linspace(0.0, z_probe_hi, 4096)
    chi_z_probe = np.asarray(
        _comoving_distance(jnp.asarray(z_probe), Omega_m, -1.0, 0.0)
    )
    W = np.ones_like(chi_nodes)
    for i in range(3):
        W = W * _fiducial_leg_weight(
            tracers[i],
            chi_nodes,
            chi_z_probe,
            z_probe,
            dndz[i],
            bias[i],
            z_source[i],
            Omega_m,
        )
    B_proxy = _bbks_pk(ell_ref / chi_nodes, Omega_m=Omega_m) ** 2
    w = np.abs(gl_weights) / chi_nodes**4 * np.abs(W) * B_proxy
    total = w.sum()
    return w / total if total > 0 else w


def _source_fraction_beyond(chi_max, z_arr, nz_arr, Omega_m=0.24):
    r"""Fraction of the source distribution lying beyond chi_max [Mpc/h].

    Uses a deliberately *low* fiducial :math:`\Omega_m` (largest
    :math:`\chi(z)` → most conservative) so the check flags any cosmology in
    a typical prior whose sources are truncated by the fixed chi_max.

    Parameters
    ----------
    chi_max : float
        Mpc/h.
    z_arr, nz_arr : (N,) ndarray
        Source n(z) table.
    Omega_m : float, optional

    Returns
    -------
    frac : float
        In [0, 1].
    """
    chi_z = np.asarray(_comoving_distance(jnp.asarray(z_arr), Omega_m, -1.0, 0.0))
    norm = _np_trapezoid(nz_arr, z_arr)
    if norm <= 0:
        return 0.0
    return float(_np_trapezoid(np.where(chi_z > chi_max, nz_arr, 0.0), z_arr) / norm)


def _cmb_fraction_beyond(chi_max, z_source, Omega_m=0.24):
    r"""Fraction of the fiducial CMB-lensing efficiency weight lying beyond
    chi_max [Mpc/h]: the ``'cmb_lensing'`` analogue of
    ``_source_fraction_beyond``. A literal :math:`\chi_\star > \chi_\mathrm{max}`
    check would almost always fire (:math:`\chi_\star` at :math:`z\sim1089`
    is far beyond the shear-tuned default chi_max=5800), so this reports a
    graded fractional weight instead.

    Parameters
    ----------
    chi_max : float
        Mpc/h.
    z_source : float
        Source-plane redshift (~1089 for the CMB).
    Omega_m : float, optional

    Returns
    -------
    frac : float
        In [0, 1].
    """
    z_probe = np.linspace(0.0, max(1200.0, 1.05 * z_source), 4096)
    chi_probe = np.asarray(_comoving_distance(jnp.asarray(z_probe), Omega_m, -1.0, 0.0))
    chi_star = float(np.interp(z_source, z_probe, chi_probe))
    chi = np.linspace(1e-6, chi_star, 4096)
    z_of_chi = np.interp(chi, chi_probe, z_probe)
    a_of_chi = 1.0 / (1.0 + z_of_chi)
    weight = chi / a_of_chi * (chi_star - chi)
    total = _np_trapezoid(weight, chi)
    beyond = _np_trapezoid(np.where(chi > chi_max, weight, 0.0), chi)
    return beyond / total if total > 0 else 0.0


def _radial_width(z_arr, nz_arr, Omega_m=0.3):
    r"""Standard deviation [Mpc/h] of :math:`\chi` under n(z), the radial width of a
    ``'counts'`` or ``'ia'`` kernel, which is proportional to n(z) H(z).

    Parameters
    ----------
    z_arr, nz_arr : (N,) ndarray
        n(z) table.
    Omega_m : float, optional

    Returns
    -------
    width : float
        Mpc/h.
    """
    chi = np.asarray(_comoving_distance(jnp.asarray(z_arr), Omega_m, -1.0, 0.0))
    norm = _np_trapezoid(nz_arr, z_arr)
    mean = _np_trapezoid(nz_arr * chi, z_arr) / norm
    return float(
        np.sqrt(max(_np_trapezoid(nz_arr * (chi - mean) ** 2, z_arr) / norm, 0.0))
    )


def warn_if_geometry_inadequate(g):
    """Warn where the fixed chi-range or the P_L k-grid does not cover `g`.

    Runs once per geometry, on first build. Emits RuntimeWarnings only --
    never raises, never affects the returned bispectrum.

    Parameters
    ----------
    g : _DiagnosticInputs

    Returns
    -------
    None
    """
    k_grid = _STANDARD_K_GRID if g.k_lin is None else np.asarray(g.k_lin)
    k_grid_min, k_grid_max = float(k_grid[0]), float(k_grid[-1])

    if g.pk_source == "baccoemu":
        # The Limber a(χ)-nodes are cosmology-dependent tracers, so they can't
        # be bounds-checked per call like make_bihalofit's expfactors.
        # Instead check at setup that the deepest node stays inside the
        # emulator's trained expfactor range even for a high-Ω_m geometry
        # (smallest a(chi_max) over a typical prior).
        lin = _get_baccoemu().emulator["linear"]
        a_lo = float(np.asarray(lin["bounds"])[list(lin["keys"]).index("expfactor")][0])
        z_probe = np.linspace(0.0, 20.0, 2048)
        chi_probe = np.asarray(
            _comoving_distance(jnp.asarray(z_probe), 0.45, -1.0, 0.0)
        )
        a_deepest = 1.0 / (1.0 + float(np.interp(g.chi_max, chi_probe, z_probe)))
        if a_deepest < a_lo:
            warnings.warn(
                f"chi_max={g.chi_max:g} Mpc/h reaches a(chi) ~ {a_deepest:.3f} for "
                f"high-Omega_m cosmologies, below the baccoemu emulator's trained "
                f"expfactor bound {a_lo}; the emulator would extrapolate silently "
                "on the deepest chi-nodes. Reduce chi_max or use pk_source="
                "'syren'/'eh'.",
                RuntimeWarning,
                stacklevel=3,
            )

    # Share of the Limber weight on (node, triangle) pairs whose k = (ell + shift)/chi leaves the P_L grid: each
    # node's fiducial weight times the share of its triangles outside. Per triangle, not the extremes of the whole set,
    # which a single near-degenerate triangle (ell3 ~ 0 in the <Map^3> grids) would push off the grid at every node.
    ells = np.stack([g.ell1, g.ell2, g.ell3]) + g.limber_shift
    ell_hi, ell_lo = ells.max(axis=0), ells.min(axis=0)
    ell_ref = float(np.exp(np.log(ells).mean()))  # representative B_δ scale
    w_fid = _fiducial_node_weights(
        g.chi_nodes, g.gl_weights, g.tracers, g.dndz, g.bias, g.z_source, ell_ref
    )
    share = np.array(
        [
            np.mean((ell_hi > k_grid_max * c) | (ell_lo < k_grid_min * c))
            for c in g.chi_nodes
        ]
    )
    out_frac = float(w_fid @ share)
    if out_frac > 1e-3:
        warnings.warn(
            f"k = (ell + {g.limber_shift:g})/chi out of P_L grid [{k_grid_min:.2e}, {k_grid_max:.2e}] h/Mpc "
            f"for {out_frac:.1%} of the Limber weight (k up to {ell_hi.max() / g.chi_nodes.min():.2e}); P_L is "
            "extrapolated there as a power law. Above k_NL BiHalofit uses P_L only in its suppressed 3-halo term, "
            "so the effect is usually small; pass a wider k_lin to remove it.",
            RuntimeWarning,
            stacklevel=3,
        )

    # A kernel proportional to n(z) is a shell: the Gauss-Legendre nodes cannot
    # resolve it once its radial width falls below their spacing,
    # pi (chi_max - chi_min) / (2 N_chi). Measured over 200 cosmologies and
    # redshift distributions, the mean error stays below 0.1% down to a width of
    # about 0.9 of the spacing and grows quickly below it.
    spacing = (
        np.pi * float(g.chi_nodes.max() - g.chi_nodes.min()) / (2 * len(g.chi_nodes))
    )
    for i in range(3):
        if g.tracers[i] in ("counts", "ia"):
            width = _radial_width(*g.dndz[i])
            if width < spacing:
                n_needed = int(
                    np.ceil(
                        np.pi
                        * float(g.chi_nodes.max() - g.chi_nodes.min())
                        / (2 * width)
                    )
                )
                warnings.warn(
                    f"leg {i}'s ({g.tracers[i]!r}) n(z) has a radial width of {width:.0f} Mpc/h, "
                    f"below the {spacing:.0f} Mpc/h spacing of the {len(g.chi_nodes)} chi-nodes; "
                    f"the projection integral is under-resolved (errors of several per cent or "
                    f"more). Use N_chi >= {n_needed}.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    # chi_max must reach the bulk of each leg's support, checked per leg on a conservative geometry.
    # A 'lensing' leg's efficiency is a trapezoidal rule on the caller's dndz grid: ~0.7% error on
    # B_kappa at 15 points, ~0.07% at 30, ~0.01% at 60.
    _DNDZ_N_MIN_WARN = 50
    for i in range(3):
        if g.tracers[i] == "lensing":
            z_arr, nz_arr = g.dndz[i]
            if len(z_arr) < _DNDZ_N_MIN_WARN:
                warnings.warn(
                    f"leg {i}'s dndz has only {len(z_arr)} points; the lensing "
                    "efficiency integral is a plain trapezoidal rule on this "
                    "exact grid, with no internal refinement, so its accuracy "
                    "is set directly by this resolution (measured: ~0.7% error "
                    "at 15 points, ~0.07% at 30, ~0.01% at 60, for a smooth "
                    "n(z)). Supply more points, or check convergence by "
                    "doubling the grid and comparing.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        if g.tracers[i] in ("lensing", "counts", "ia"):
            z_arr, nz_arr = g.dndz[i]
            src_beyond = _source_fraction_beyond(g.chi_max, z_arr, nz_arr)
            if src_beyond > 1e-3:
                warnings.warn(
                    f"chi_max={g.chi_max:g} Mpc/h truncates ~{src_beyond:.1%} of "
                    f"leg {i}'s ({g.tracers[i]!r}) source distribution for "
                    "low-Ω_m cosmologies; increase chi_max.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        else:  # 'cmb_lensing'
            # The integrand is the product of the three kernels, so it vanishes
            # beyond the earliest end of any leg's support: this leg's own
            # truncation is harmless while another leg's sources end inside chi_max.
            bounded = any(
                g.tracers[j] in ("lensing", "counts", "ia")
                and _source_fraction_beyond(g.chi_max, *g.dndz[j]) <= 1e-3
                for j in range(3)
                if j != i
            )
            cmb_beyond = _cmb_fraction_beyond(g.chi_max, g.z_source[i])
            if cmb_beyond > 1e-3 and not bounded:
                warnings.warn(
                    f"chi_max={g.chi_max:g} Mpc/h truncates ~{cmb_beyond:.1%} of "
                    f"leg {i}'s cmb_lensing efficiency weight "
                    f"(z_source={g.z_source[i]:g}); increase chi_max.",
                    RuntimeWarning,
                    stacklevel=3,
                )
