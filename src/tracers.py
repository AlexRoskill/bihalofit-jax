"""Per-leg projection kernels for the Limber-projected bispectrum.

  'lensing'      weak-lensing efficiency
  'counts'       galaxy clustering (no magnification)
  'cmb_lensing'  lensing of a single source plane at z_source
  'ia'           NLA intrinsic alignments: 'counts' with an amplitude f_IA(z)

Intrinsic-alignment terms (GGI, GII, III) are combinations of 'lensing' and 'ia' legs, summed by the
caller. `make_tracer_kernel` binds a leg's static configuration and returns

    W = tracer_kernel(Omega_m, w0, wa, chi_bg, a_nodes, z_nodes, nz, bias, ia)

The n(z) values and bias are traced, so they can be differentiated; the z grid is static. n(z) is
normalised inside the trace, so a parameter that only rescales it has zero gradient.
"""

import numpy as np
import jax.numpy as jnp

from .background import (
    _C_OVER_H0,
    _RAMP_EPS,
    _E2_of_a,
    _comoving_distance,
    lensing_geometry,
    _smooth_lookup,
    _pad_grid_for_lookup,
    _trapz_last,
)

__all__ = [
    "lensing_kernel",
    "counts_kernel",
    "cmb_lensing_kernel",
    "ia_kernel",
    "ia_amplitude",
    "make_tracer_kernel",
    "validate_tracer_config",
]

_VALID_TRACERS = ("lensing", "counts", "cmb_lensing", "ia")

# NLA amplitude constant C_1 * rho_crit (dimensionless) and the default pivot
# redshift of the power-law evolution. See ia_amplitude.
_C1_RHO_CRIT = 0.0134
_IA_Z_PIVOT_DEFAULT = 0.62


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def lensing_kernel(Omega_m, w0, wa, chi_nodes, a_nodes, z_src, nz_src):
    r"""Weak-lensing efficiency kernel.

    .. math::
        W(\chi) = \frac{3}{2}\,\Omega_m \left(\frac{H_0}{c}\right)^2
                  \frac{\chi}{a(\chi)}\, g(\chi)

    with :math:`g` the lensing efficiency integral over the source n(z).

    Parameters
    ----------
    Omega_m, w0, wa : scalar
        Total matter density today and CPL dark-energy parameters.
    chi_nodes, a_nodes : (N_chi,)
        Comoving distances (Mpc/h) and scale factors of the nodes.
    z_src, nz_src : (N_z,)
        Source redshift grid and n(z), normalised to unit integral.

    Returns
    -------
    W : (N_chi,)
        In h/Mpc.
    """
    chi_z = _comoving_distance(z_src, Omega_m, w0, wa)
    g = lensing_geometry(chi_nodes, chi_z, nz_src, z_src)
    return 1.5 / _C_OVER_H0**2 * Omega_m * (chi_nodes / a_nodes * g)


def counts_kernel(Omega_m, w0, wa, bias, z_nodes, z_tab, nz_tab, n_tab):
    r"""Galaxy-clustering kernel, :math:`W(\chi) = b\, n_g(z)\, H(z)/c`.

    Parameters
    ----------
    Omega_m, w0, wa : scalar
    bias : scalar or (N_chi,)
        Linear galaxy bias (``ia_kernel`` passes f_IA here).
    z_nodes : (N_chi,)
        Redshifts of the chi-nodes.
    z_tab, nz_tab : (n_tab,)
        Padded, normalised n(z) table.
    n_tab : int
        Length of the table, static.

    Returns
    -------
    W : (N_chi,)
        In h/Mpc.
    """
    ng = _smooth_lookup(z_tab, nz_tab, z_nodes, n_tab)
    E = jnp.sqrt(_E2_of_a(1.0 / (1.0 + z_nodes), Omega_m, w0, wa))
    return bias * ng * E / _C_OVER_H0


def ia_amplitude(Omega_m, A_IA, eta, z_pivot, D, z):
    r"""NLA intrinsic-alignment amplitude :math:`f_\mathrm{IA}(z)`.

    .. math::
        f_\mathrm{IA}(z) = -A_\mathrm{IA}\, C_1\rho_\mathrm{crit}\,
            \frac{\Omega_m}{D(z)}
            \left[\frac{1+z}{1+z_\mathrm{pivot}}\right]^{\eta}

    :math:`\Omega_m` is the matter density *today*, and :math:`D(z)` the linear
    growth factor with :math:`D(0)=1`. The density factor is constant because
    the physical mean density :math:`\bar\rho(z)\propto a^{-3}` cancels against
    the :math:`a^2` of the comoving Poisson equation and the :math:`(1+z)` inside
    Hirata & Seljak's rescaled growth factor (their Eqs. 13-14), leaving
    :math:`\bar\rho_0/D(z)`. ``eta = 0`` is plain NLA.

    Parameters
    ----------
    Omega_m, A_IA, eta, z_pivot : scalar
    D, z : array
        Growth factor (D(0)=1) and redshift, same shape.

    Returns
    -------
    f_IA : shape of z
    """
    return -A_IA * _C1_RHO_CRIT * Omega_m / D * ((1.0 + z) / (1.0 + z_pivot)) ** eta


def ia_kernel(
    Omega_m, w0, wa, A_IA, eta, z_pivot, D_nodes, z_nodes, z_tab, nz_tab, n_tab
):
    r"""Intrinsic-alignment kernel, :math:`W(\chi) = f_\mathrm{IA}(z)\, n(z)\, H(z)/c`.

    The galaxy-clustering kernel with the bias replaced by the array
    :math:`f_\mathrm{IA}(z(\chi))`. ``D_nodes`` is the growth factor at the
    chi-nodes. The sign follows :func:`ia_amplitude`: it is negative for
    ``A_IA > 0``, and the caller must not drop negative weights.

    Parameters
    ----------
    Omega_m, w0, wa, A_IA, eta, z_pivot : scalar
    D_nodes, z_nodes : (N_chi,)
    z_tab, nz_tab : (n_tab,)
        Padded, normalised n(z) table.
    n_tab : int

    Returns
    -------
    W : (N_chi,)
        In h/Mpc; negative for ``A_IA > 0``.
    """
    f_IA = ia_amplitude(Omega_m, A_IA, eta, z_pivot, D_nodes, z_nodes)
    return counts_kernel(Omega_m, w0, wa, f_IA, z_nodes, z_tab, nz_tab, n_tab)


def cmb_lensing_kernel(Omega_m, chi_nodes, a_nodes, chi_star):
    r"""Lensing kernel for a single source plane at :math:`\chi_\star`.

    The same form as :func:`lensing_kernel` with the n(z) integral replaced by
    the smoothed ramp :math:`(\chi_\star - \chi)/\chi_\star`.

    Parameters
    ----------
    Omega_m : scalar
    chi_nodes, a_nodes : (N_chi,)
    chi_star : scalar
        Comoving distance to the source plane, Mpc/h.

    Returns
    -------
    W : (N_chi,)
        In h/Mpc.
    """
    x = chi_star - chi_nodes
    ramp = 0.5 * (x + jnp.sqrt(x * x + _RAMP_EPS**2))
    return 1.5 / _C_OVER_H0**2 * Omega_m * (chi_nodes / a_nodes) * (ramp / chi_star)


# ---------------------------------------------------------------------------
# Per-leg binding
# ---------------------------------------------------------------------------


def _padded_n_of_z(dndz_i, z_bg, jdtype):
    """Static pieces a 'counts' or 'ia' leg needs to look n(z) up at the chi-nodes.

    Padded to the background z-range first, so the lookup never has to
    extrapolate off the end of the caller's own n(z) grid. The grid and the pad
    counts are static; only the values are padded in-trace, by ``table(nz)``,
    which also normalises them.

    Parameters
    ----------
    dndz_i : (z_arr, nz_arr) tuple
        The leg's n(z) table.
    z_bg : (N_zbg,) ndarray
        Background z-grid the table is padded to.
    jdtype : JAX dtype

    Returns
    -------
    z_pad : (n_tab,) jax array
        Padded z-grid.
    n_tab : int
    table : callable
        ``table(nz)``: (len(z_arr),) n(z) values -> (n_tab,) normalised, zero-padded values.
    """
    z_pad, n_lo, n_hi = _pad_grid_for_lookup(dndz_i[0], z_bg[0], z_bg[-1])
    z_raw = jnp.asarray(dndz_i[0], dtype=jdtype)
    z_j = jnp.asarray(z_pad, dtype=jdtype)
    pad_lo = jnp.zeros((n_lo,), dtype=jdtype)
    pad_hi = jnp.zeros((n_hi,), dtype=jdtype)

    def table(nz):
        """(len(z_arr),) n(z) values -> (n_tab,) normalised and zero-padded."""
        nz_n = nz / _trapz_last(nz, z_raw)
        return jnp.concatenate([pad_lo, nz_n, pad_hi])

    return z_j, len(z_pad), table


def make_tracer_kernel(
    tracer, dndz_i, z_source_i, chi_nodes_j, z_bg, jdtype, z_pivot=_IA_Z_PIVOT_DEFAULT
):
    """Bind one leg's static configuration; return W_i(Omega_m, w0, wa, ...).

    The returned closure takes ``ia = (A_IA, eta, D_nodes)`` as its last
    argument -- the alignment parameters and the growth factor at the
    chi-nodes -- and ignores it unless the leg is an 'ia' leg.

    Parameters
    ----------
    tracer : {'lensing', 'counts', 'ia', 'cmb_lensing'}
    dndz_i : (z_arr, nz_arr) tuple or None
        The leg's n(z) table; None for 'cmb_lensing'.
    z_source_i : float or None
        Source-plane redshift, used for 'cmb_lensing'.
    chi_nodes_j : (N_chi,) jax array
        Limber nodes in Mpc/h.
    z_bg : (N_zbg,) ndarray
        Shared background z-grid.
    jdtype : JAX dtype
    z_pivot : float, optional
        IA pivot redshift.

    Returns
    -------
    tracer_kernel : callable
        ``tracer_kernel(Omega_m, w0, wa, chi_bg, a_nodes, z_nodes, nz, bias, ia)`` -> W, (N_chi,) in
        h/Mpc. Cosmology arguments are scalars, ``chi_bg`` is chi on ``z_bg``, ``a_nodes`` and
        ``z_nodes`` are (N_chi,), ``nz`` the leg's n(z) values; arguments the leg does not use are
        ignored.
    """
    if tracer == "lensing":
        # bias and ia are unused here; the signature is uniform across tracers.
        z_j = jnp.asarray(dndz_i[0], dtype=jdtype)

        def tracer_kernel(Omega_m, w0, wa, chi_bg, a_nodes, z_nodes, nz, bias, ia):
            """Lensing W, (N_chi,); see make_tracer_kernel for the arguments."""
            nz_n = nz / _trapz_last(nz, z_j)
            return lensing_kernel(Omega_m, w0, wa, chi_nodes_j, a_nodes, z_j, nz_n)

        return tracer_kernel

    if tracer == "counts":
        z_j, n_tab, table = _padded_n_of_z(dndz_i, z_bg, jdtype)

        def tracer_kernel(Omega_m, w0, wa, chi_bg, a_nodes, z_nodes, nz, bias, ia):
            """Counts W, (N_chi,); see make_tracer_kernel for the arguments."""
            return counts_kernel(Omega_m, w0, wa, bias, z_nodes, z_j, table(nz), n_tab)

        return tracer_kernel

    if tracer == "ia":
        z_j, n_tab, table = _padded_n_of_z(dndz_i, z_bg, jdtype)

        def tracer_kernel(Omega_m, w0, wa, chi_bg, a_nodes, z_nodes, nz, bias, ia):
            """IA W, (N_chi,); ``ia = (A_IA, eta, D_nodes)``. See make_tracer_kernel."""
            A_IA, eta, D_nodes = ia
            return ia_kernel(
                Omega_m,
                w0,
                wa,
                A_IA,
                eta,
                z_pivot,
                D_nodes,
                z_nodes,
                z_j,
                table(nz),
                n_tab,
            )

        return tracer_kernel

    # 'cmb_lensing': z_bg and z_source are both static, so chi_star reduces to
    # a fixed two-point interpolation of the cosmology-dependent chi_bg.
    j_hi = int(np.clip(np.searchsorted(z_bg, z_source_i), 1, len(z_bg) - 1))
    j_lo = j_hi - 1
    frac = float((z_source_i - z_bg[j_lo]) / (z_bg[j_hi] - z_bg[j_lo]))

    def tracer_kernel(Omega_m, w0, wa, chi_bg, a_nodes, z_nodes, nz, bias, ia):
        """CMB-lensing W, (N_chi,); see make_tracer_kernel for the arguments."""
        # nz/bias/ia are unused (no source distribution, no clustering): the
        # signature is uniform so the Limber kernel never branches on tracer type.
        chi_star = (1.0 - frac) * chi_bg[j_lo] + frac * chi_bg[j_hi]
        return cmb_lensing_kernel(Omega_m, chi_nodes_j, a_nodes, chi_star)

    return tracer_kernel


def validate_tracer_config(dndz, tracers, bias, z_source):
    """Check the per-leg tracer configuration, raising on the first problem.

    dndz is required exactly where a leg is 'lensing', 'counts' or 'ia', and bias
    exactly where it is 'counts'; both must be absent otherwise. Rejecting an
    unused value rather than ignoring it is deliberate -- a bias passed
    alongside a 'lensing' leg means the caller expected it to do something.

    Parameters
    ----------
    dndz, tracers, bias, z_source : length-3 tuples
        Per-leg (z_arr, nz_arr) or None, tracer name, bias or None, source redshift.

    Returns
    -------
    None
        Raises ValueError.
    """
    if len(dndz) != 3 or len(tracers) != 3 or len(bias) != 3 or len(z_source) != 3:
        raise ValueError(
            "dndz, tracers, bias, z_source must each be length-3 tuples "
            "(one entry per triangle leg)."
        )

    for i, tr in enumerate(tracers):
        if tr not in _VALID_TRACERS:
            raise ValueError(
                f"tracers[{i}] must be one of {_VALID_TRACERS} (got {tr!r})."
            )
        needs_dndz = tr in ("lensing", "counts", "ia")
        if needs_dndz and dndz[i] is None:
            raise ValueError(f"dndz[{i}] is required for tracers[{i}]={tr!r}.")
        if not needs_dndz and dndz[i] is not None:
            raise ValueError(f"dndz[{i}] must be None for tracers[{i}]={tr!r}.")
        if tr == "counts" and bias[i] is None:
            raise ValueError(f"bias[{i}] is required for tracers[{i}]='counts'.")
        if tr != "counts" and bias[i] is not None:
            raise ValueError(f"bias[{i}] must be None for tracers[{i}]={tr!r}.")
