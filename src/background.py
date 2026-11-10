"""
Flat-wCDM background geometry: chi(z), E(z), and the lensing efficiency
kernel, plus the small grid helpers they need.

Everything here is cosmology-only -- no bispectrum, no Limber projection, no
tracer configuration -- so it can be used and tested on its own. The
CCL cross-check suite does exactly that.

All distances are in Mpc/h and c/H0 = 2998 Mpc/h is h-independent, so no
factor of h appears anywhere in this module.
"""

import numpy as np
import jax
import jax.numpy as jnp

from ._utils import _crom_eval, _c1_newton_invert

# c/H0 in Mpc/h -- independent of h (c/H0 = 2998 Mpc/h for any h)
_C_OVER_H0 = 2998.0  # Mpc/h

# Width of the smooth-ReLU corner used by every lensing ramp -- the integrated
# one in lensing_geometry below and the single-source one in tracers.py, which
# imports this so the two cannot drift apart. See lensing_geometry for why the
# ramp is softened at all.
_RAMP_EPS = 2.0  # Mpc/h


def _cumtrapz0(y, x):
    """Cumulative trapezoid of y(x) with a leading 0, same length as x (JAX).

    Parameters
    ----------
    y, x : (N,)

    Returns
    -------
    cum : (N,)
        ``cum[i] = int_{x[0]}^{x[i]} y dx``.
    """
    seg = 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
    return jnp.concatenate([jnp.zeros((1,), dtype=y.dtype), jnp.cumsum(seg)])


def _trapz_last(y, x):
    """Trapezoidal integral of y over x along the last axis (x is 1-D, JAX).

    Parameters
    ----------
    y : (..., N)
    x : (N,)

    Returns
    -------
    I : (...)
    """
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)


def _E2_of_a(a, Omega_m, w0, wa):
    r"""Flat-wCDM :math:`E^2(a) = (H/H_0)^2` (JAX).

    Omega_m is the dimensionless total matter density; dark energy follows the
    CPL form :math:`w(a) = w_0 + w_a (1 - a)`. This is the single definition
    of the background expansion rate; take ``jnp.sqrt`` of it where E itself
    is wanted.

    Parameters
    ----------
    a : array
        Scale factor.
    Omega_m, w0, wa : scalar or array broadcastable with a

    Returns
    -------
    E2 : broadcast shape of the inputs
    """
    de = a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - a))
    return Omega_m * a**-3.0 + (1.0 - Omega_m) * de


def _comoving_distance(z_grid, Omega_m, w0, wa):
    r""":math:`\chi(z)` [Mpc/h] on z_grid for flat wCDM (JAX, differentiable).

    Parameters
    ----------
    z_grid : (N_z,)
        Increasing redshifts starting at 0; the integral is a trapezoid sum on this grid.
    Omega_m : scalar
        DIMENSIONLESS total matter density parameter.
    w0, wa : scalar
        CPL dark-energy parameters.

    Returns
    -------
    chi : (N_z,)
        Mpc/h, with ``chi[0] = 0`` when ``z_grid[0] = 0``.
    """
    E_z = jnp.sqrt(_E2_of_a(1.0 / (1.0 + z_grid), Omega_m, w0, wa))
    return _C_OVER_H0 * _cumtrapz0(1.0 / E_z, z_grid)


def _omega_de_of_a(a, Omega_m, w0, wa):
    r"""Flat-wCDM dark-energy density fraction :math:`\Omega_{de}(a)` (JAX).

    Reuses :func:`_E2_of_a` for :math:`E^2(a)` -- this is the same background
    the rest of the module (chi(z), etc.) is built on.

    Parameters
    ----------
    a : array
        Scale factor.
    Omega_m, w0, wa : scalar or array broadcastable with a

    Returns
    -------
    Omega_de : broadcast shape of the inputs
    """
    de = a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - a))
    return (1.0 - Omega_m) * de / _E2_of_a(a, Omega_m, w0, wa)


# ---------------------------------------------------------------------------
# Linear growth factor D(a) -- fixed-step RK4 ODE solve, not a closed-form
# quadrature. See docstring of _growth_D_unnorm for why a closed-form
# shortcut is wrong for wCDM/CPL.
# ---------------------------------------------------------------------------

_GROWTH_ODE_A0 = 1e-4  # start deep in matter domination (z=9999)
_GROWTH_ODE_N_STEPS = 128  # RK4 steps from a0 to the target a -- gives
# ~1e-7 relative accuracy vs Takahashi's C code
# (checked at n_steps=128 vs 512, wCDM w0=-0.8),
# well below the ~2.5e-5 float32 precision floor
# this package normally runs at; more steps
# would just cost speed for no real gain


def _growth_D_unnorm(
    a_scalar, Omega_m, w0, wa, jdtype, a0=_GROWTH_ODE_A0, n_steps=_GROWTH_ODE_N_STEPS
):
    r"""Unnormalised linear growth factor D(a) for flat wCDM/CPL dark energy
    (w(a) = w0 + wa*(1-a)), via fixed-step RK4 integration of the standard
    smooth-dark-energy growth equation in x = ln a, solved for g = D/a
    (well-conditioned: g -> 1 deep in matter domination, unlike D itself
    which -> 0):

      g'' + [5/2 - (3/2) w(a) Omega_de(a)] g' + (3/2)(1-w(a)) Omega_de(a) g = 0

    This is the same equation Takahashi+2020's bihalofit reference C code
    integrates (its `lgr`/`lgr_func`, RK4 with step-doubling) -- verified
    here to agree with that code to ~1e-6 relative for LambdaCDM and ~1e-6
    for wCDM (checked at w0 in {-0.8, -1.2}, wa=0) via a compiled ctypes
    cross-check.

    The closed-form Heath (1977) integral is not used: it solves this equation only for
    w = -1 (or -1/3), and is off by 3-6% at z ~ 1-5 for w0 != -1.

    a_scalar : scalar (may be a JAX tracer, e.g. from Limber a(chi)).
    Omega_m, w0, wa : JAX arrays of shape (N_cosmo,).
    Returns unnormalised D(a), shape (N_cosmo,).
    """
    a_s = jnp.asarray(a_scalar, dtype=jdtype)

    def dydx(x, y):
        """Right-hand side at x = ln a: y = (g, g'), each (N_cosmo,) -> (g', g''), stacked (2, N_cosmo)."""
        g, gp = y
        a = jnp.exp(x)
        Ode_a = _omega_de_of_a(a, Omega_m, w0, wa)
        w_a = w0 + wa * (1.0 - a)
        gpp = -(2.5 - 1.5 * w_a * Ode_a) * gp - 1.5 * (1.0 - w_a) * Ode_a * g
        return jnp.stack([gp, gpp])

    x0 = jnp.log(jnp.asarray(a0, dtype=jdtype))
    x1 = jnp.log(a_s)
    h = (x1 - x0) / n_steps

    def rk4_step(carry, _):
        """One RK4 step for lax.scan: carry (x, y) -> ((x + h, y_new), None)."""
        x, y = carry
        k1 = dydx(x, y)
        k2 = dydx(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = dydx(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = dydx(x + h, y + h * k3)
        y_new = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return (x + h, y_new), None

    y0 = jnp.stack([jnp.ones_like(Omega_m), jnp.zeros_like(Omega_m)])  # g=1, g'=0
    (_, y_final), _ = jax.lax.scan(rk4_step, (x0, y0), None, length=n_steps)
    return a_s * y_final[0]


def growth_ratios(a_sorted, Omega_m, w0, wa, jdtype):
    """D(a)/D(1) for every a in a_sorted.

    Parameters
    ----------
    a_sorted : (N_z,)
        Scale factors (numpy or JAX tracer).
    Omega_m, w0, wa : (N_cosmo,)
        Total matter density and CPL dark-energy parameters.
    jdtype : JAX dtype

    Returns
    -------
    D_ratio : (N_z, N_cosmo)
    """

    def D(a):
        """Unnormalised D at scalar a, (N_cosmo,)."""
        return _growth_D_unnorm(a, Omega_m, w0, wa, jdtype)

    D_z0 = D(jnp.asarray(1.0, dtype=jdtype))
    return jax.vmap(lambda a: D(a) / D_z0)(jnp.asarray(a_sorted, dtype=jdtype))


def fiducial_background(z_max, N_z, Omega_m, w0=-1.0, wa=0.0):
    r"""Flat-wCDM comoving distance :math:`\chi(z)` [Mpc/h]; numpy, for diagnostics.

    Omega_m is the DIMENSIONLESS total matter density parameter :math:`\Omega_m`
    (NOT :math:`\omega = \Omega h^2`).

    Parameters
    ----------
    z_max : float
        Upper end of the uniform z-grid.
    N_z : int
        Number of grid points.
    Omega_m, w0, wa : float

    Returns
    -------
    z, chi : (N_z,) ndarray
        Redshifts and comoving distances in Mpc/h, with chi[0] = 0.
    """
    z = np.linspace(0.0, float(z_max), int(N_z))
    chi = np.asarray(_comoving_distance(jnp.asarray(z), Omega_m, w0, wa))
    return z, chi


def lensing_geometry(chi_nodes, chi_z, nz_grid, z_grid):
    r"""Redshift-weighted lensing efficiency :math:`g(\chi_j)` (JAX).

    .. math::
        g(\chi_j) = \int_{z(\chi_j)}^{z_H} n(z')\,
            \frac{\chi(z') - \chi_j}{\chi(z')}\, dz'

    Parameters
    ----------
    chi_nodes : (N_chi,) GL nodes [Mpc/h]   (concrete)
    chi_z     : (N_z,)   :math:`\chi`\ (z_grid) [Mpc/h]  (JAX tracer, per cosmology)
    nz_grid   : (N_z,)   normalized n(z) on z_grid
    z_grid    : (N_z,)   redshift grid (concrete; chi_z[0] corresponds to z=0)

    Returns
    -------
    g : (N_chi,)
        Lensing efficiency at each chi-node (dimensionless).

    Notes
    -----
    The lower limit :math:`z' > z(\chi_j)` is enforced smoothly by the
    :math:`(\chi(z') - \chi_j)` ramp, which is exactly zero for
    :math:`\chi(z') \le \chi_j`, continuous and differentiable almost
    everywhere.
    """
    # Smooth ReLU, not a hard clip: the hard ramp kinks g(χ) wherever
    # χ(z')=χ_node as distances move across z-grid nodes, which kinks the
    # gradient and makes NUTS diverge. 0.5(x+√(x²+ε²)) is C^∞ and matches the
    # ramp to O(ε); _RAMP_EPS=2 Mpc/h ≪ the ~10³ Mpc/h kernel scale.
    x = chi_z[None, :] - chi_nodes[:, None]  # (N_chi, N_z)
    ramp = 0.5 * (x + jnp.sqrt(x * x + _RAMP_EPS**2))  # (N_chi, N_z)
    denom = jnp.where(chi_z > 0.0, chi_z, 1.0)  # avoid 0/0 at z=0
    integrand = nz_grid[None, :] * ramp / denom[None, :]  # (N_chi, N_z)
    return _trapz_last(integrand, z_grid)  # (N_chi,)


def _smooth_lookup(x_grid_j, y_grid_j, x_query, N):
    r"""C^1 lookup of y at x_query from the table (x_grid_j, y_grid_j) (JAX).

    Finds the continuous index where the Catmull-Rom interpolant of x_grid_j equals
    x_query, then evaluates the Catmull-Rom interpolant of y_grid_j there. Unlike
    ``jnp.interp``, the result stays C^1 as cosmology moves x_grid_j or x_query.

    Parameters
    ----------
    x_grid_j : (N,)
        Monotonically increasing abscissae: chi(z) on the background z-grid [Mpc/h]
        for the chi -> z inversion, or the zero-padded n(z) redshift grid.
    y_grid_j : (N,)
        Values on x_grid_j: the background z-grid, or the normalised, padded n(z).
    x_query : (M,)
        Points to look up: the chi-nodes [Mpc/h], or the node redshifts z(chi_j).
    N : int
        len(x_grid_j), static.

    Returns
    -------
    y : (M,)
        y at x_query: z(chi_j), or n(z(chi_j)).
    """
    idx0 = jnp.clip(jnp.searchsorted(x_grid_j, x_query), 1, N - 1)
    x_lo, x_hi = x_grid_j[idx0 - 1], x_grid_j[idx0]
    xi_seed = (idx0 - 1) + (x_query - x_lo) / (x_hi - x_lo)
    xi = _c1_newton_invert(x_grid_j, x_query, xi_seed, N)
    y, _ = _crom_eval(y_grid_j, xi)
    return y


def _background_zmax(chi_max, tracers, z_source, Omega_m_probe=0.45):
    r"""Conservative z_max for the shared background z-grid (numpy, setup-time).

    High enough that :math:`\chi(z_\mathrm{max})` exceeds chi_max for any
    cosmology in a typical prior, and high enough to cover any
    ``'cmb_lensing'`` leg's ``z_source``. :math:`\Omega_m=0.45` is a
    conservative (high) choice since larger :math:`\Omega_m` needs a larger
    z to reach the same chi_max.

    Parameters
    ----------
    chi_max : float
        Mpc/h.
    tracers : sequence of str
        Per-leg tracer names.
    z_source : sequence of float
        Per-leg source redshifts (used for 'cmb_lensing' legs).
    Omega_m_probe : float, optional

    Returns
    -------
    z_max : float
        At least 4.
    """
    z_probe = np.linspace(0.0, 20.0, 4096)
    chi_probe = np.asarray(
        _comoving_distance(jnp.asarray(z_probe), Omega_m_probe, -1.0, 0.0)
    )
    z_needed = float(np.interp(chi_max, chi_probe, z_probe))
    for tr, zs in zip(tracers, z_source):
        if tr == "cmb_lensing":
            z_needed = max(z_needed, float(zs))
    return max(4.0, 1.05 * z_needed)


def _background_zgrid(z_max, N, z_split=10.0, N_tail=500):
    r"""Shared background z-grid (numpy, setup-time).

    Uniform with ``N`` points on :math:`[0, z_\mathrm{split}]` (or up to
    ``z_max`` when that is smaller), then log-spaced with ``N_tail`` points out
    to ``z_max``. :math:`\chi(z)` is a trapezoid sum on this grid, so a single
    uniform grid stretched to a ``'cmb_lensing'`` leg's :math:`z=1089` would
    leave only a handful of points below :math:`z=3`, where :math:`1/E(z)`
    changes fastest, and the result would be up to ~25% off. When ``z_max`` is at
    most ``z_split`` (any shear or clustering configuration) this is exactly
    ``np.linspace(0, z_max, N)``.

    Parameters
    ----------
    z_max : float
    N : int
        Points on the uniform segment.
    z_split : float, optional
        Where the log-spaced tail starts.
    N_tail : int, optional
        Points in the tail.

    Returns
    -------
    z : (N,) or (N + N_tail,) ndarray
        Increasing, starting at 0.
    """
    if z_max <= z_split:
        return np.linspace(0.0, z_max, int(N))
    return np.concatenate(
        [
            np.linspace(0.0, z_split, int(N)),
            np.geomspace(z_split, z_max, int(N_tail) + 1)[1:],
        ]
    )


def _pad_grid_for_lookup(z_arr, z_lo, z_hi, eps=1e-3):
    r"""Zero-padding plan for a z-grid: the padded grid and how many points
    were prepended/appended (numpy, setup-time).

    Split out from :func:`_pad_dndz_for_lookup` so the traced path can pad
    n(z) *values* on device while the grid and the pad counts -- both of
    which must be static -- are decided here, once.

    Parameters
    ----------
    z_arr : (N,) array-like
        Increasing redshift grid.
    z_lo, z_hi : float
        Range to cover.
    eps : float, optional
        Gap between the table edge and the first zero.

    Returns
    -------
    z_pad : (N + n_lo + n_hi,) ndarray
    n_lo, n_hi : int
        Points prepended and appended (0 or 2 each).
    """
    z_arr = np.asarray(z_arr, dtype=float)
    n_lo = n_hi = 0
    if z_arr[0] > z_lo:
        z_arr = np.concatenate([[z_lo, max(z_lo, z_arr[0] - eps)], z_arr])
        n_lo = 2
    if z_arr[-1] < z_hi:
        z_arr = np.concatenate([z_arr, [z_arr[-1] + eps, z_hi]])
        n_hi = 2
    return z_arr, n_lo, n_hi


def _pad_dndz_for_lookup(z_arr, nz_arr, z_lo, z_hi, eps=1e-3):
    r"""Zero-pad a (z_arr, nz_arr) table out to [z_lo, z_hi] (numpy, setup-time).

    Only needed for ``'counts'`` legs: ``_smooth_lookup`` queries
    :math:`n_g(z_\mathrm{node})` over the entire background z-range, not
    just the tomographic bin's own support, so an unpadded table would
    extrapolate a nonzero "density" past its real support. ``'lensing'``
    doesn't need this: ``lensing_geometry``'s own integral already vanishes
    outside the n(z) support.

    Parameters
    ----------
    z_arr, nz_arr : (N,) array-like
        n(z) table.
    z_lo, z_hi : float
        Range to cover.
    eps : float, optional

    Returns
    -------
    z_pad, nz_pad : (N + n_lo + n_hi,) ndarray
        Grid and n(z) with zeros in the padding.
    """
    z_pad, n_lo, n_hi = _pad_grid_for_lookup(z_arr, z_lo, z_hi, eps)
    nz_pad = np.concatenate(
        [np.zeros(n_lo), np.asarray(nz_arr, dtype=float), np.zeros(n_hi)]
    )
    return z_pad, nz_pad
