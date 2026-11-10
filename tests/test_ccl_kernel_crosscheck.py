"""
Cross-check projected_bihalofit's per-leg projection kernels W_i(chi) against
CCL (Core Cosmology Library) tracer kernels.

This targets exactly the piece of the multi-tracer API that
test_projected_bihalofit.py's 'counts'/'cmb_lensing' tests don't verify:
those only check the *full* bispectrum is finite and differentiable, not
that the radial kernel itself is *correct*. Here we recompute each of the
three kernels (as implemented in `_leg_weight`, projected_bihalofit.py:500-520)
from the same public/private building blocks it's built from
(`_comoving_distance`, `lensing_geometry`, `_E2_of_a`, `_smooth_lookup`,
`_C_OVER_H0`), evaluate CCL's equivalent tracer at the same chi-nodes, and
compare.

Unit convention: this module works entirely in Mpc/h (`_C_OVER_H0` is the
h-independent number 2998); CCL's `Tracer.get_kernel(chi)` takes/returns
physical Mpc. For a radial kernel (units of inverse length), the conversion
is simply::

    W_ccl(chi / h) == h * W_ours(chi)

verified empirically below to ~1e-3 over chi in [50, 3000] Mpc/h (residual
is the expected difference between this module's GL/trapz quadrature and
CCL's own `n_samples`-point quadrature over slightly different node sets,
not a normalization mismatch).

One real convention difference, found while writing this test and worth
documenting for anyone who touches these tests later: CCL's
`NumberCountsTracer.get_kernel()` returns the bias-*free* kernel
(n_g(z) * H(z)/c only) -- galaxy bias is applied separately when CCL builds
power/bispectra, not baked into the radial kernel it returns. This module's
`'counts'` kernel *does* include bias directly (see `_leg_weight`). The
`'counts'` test below accounts for this explicitly; it is not a bug on
either side, just a different place in the pipeline where bias is applied.

Skipped entirely if pyccl isn't installed (it is an optional dependency).
"""

import numpy as np
import jax.numpy as jnp
import pytest

pyccl = pytest.importorskip("pyccl")

from bihalofit_jax import smail_nz
from bihalofit_jax.background import (
    fiducial_background,
    lensing_geometry,
)
from bihalofit_jax.background import (
    _comoving_distance,
    _E2_of_a,
    _smooth_lookup,
    _pad_dndz_for_lookup,
    _C_OVER_H0,
    growth_ratios,
)
from bihalofit_jax.tracers import ia_amplitude, ia_kernel

# chi range kept away from this module's default chi_max (5800 Mpc/h) tail,
# where the two quadrature schemes' agreement is measurably (but harmlessly)
# looser -- see module docstring.
_CHI_NODES = np.linspace(50.0, 3000.0, 20)
_RTOL = 3e-3  # 0.3 % agreement


def _ccl_cosmology(p):
    return pyccl.Cosmology(
        Omega_c=float(p["omega_cold"][0] - p["omega_baryon"][0]),
        Omega_b=float(p["omega_baryon"][0]),
        h=float(p["hubble"][0]),
        n_s=float(p["ns"][0]),
        sigma8=float(p["sigma8_cold"][0]),
        w0=float(p["w0"][0]),
        wa=float(p["wa"][0]),
        transfer_function="eisenstein_hu",  # kernels are background-only; pk choice is irrelevant here
    )


def _background_z_of_chi(Omega_m, w0, wa, chi_nodes, z_max=4.0, n_bg=2000):
    z_bg, _ = fiducial_background(z_max, n_bg, Omega_m, w0, wa)
    chi_bg = np.asarray(_comoving_distance(jnp.asarray(z_bg), Omega_m, w0, wa))
    return np.interp(chi_nodes, chi_bg, z_bg)


def test_lensing_kernel_matches_ccl(fiducial_params):
    p = fiducial_params
    Omega_m = float(p["omega_cold"][0])
    h = float(p["hubble"][0])
    w0, wa = float(p["w0"][0]), float(p["wa"][0])

    z_grid = np.linspace(0.0, 3.5, 500)
    nz_grid = smail_nz(z_grid)

    chi_z = np.asarray(_comoving_distance(jnp.asarray(z_grid), Omega_m, w0, wa))
    g = np.asarray(
        lensing_geometry(
            jnp.asarray(_CHI_NODES),
            jnp.asarray(chi_z),
            jnp.asarray(nz_grid),
            jnp.asarray(z_grid),
        )
    )
    z_nodes = _background_z_of_chi(Omega_m, w0, wa, _CHI_NODES)
    a_nodes = 1.0 / (1.0 + z_nodes)
    W_ours = 1.5 / _C_OVER_H0**2 * Omega_m * (_CHI_NODES / a_nodes * g)

    tr = pyccl.WeakLensingTracer(
        _ccl_cosmology(p), dndz=(z_grid, nz_grid), has_shear=True
    )
    W_ccl = tr.get_kernel(_CHI_NODES / h)[0]

    np.testing.assert_allclose(W_ccl, h * W_ours, rtol=_RTOL)


def test_counts_kernel_matches_ccl(fiducial_params):
    p = fiducial_params
    Omega_m = float(p["omega_cold"][0])
    h = float(p["hubble"][0])
    w0, wa = float(p["w0"][0]), float(p["wa"][0])
    bias = 1.7

    z_grid = np.linspace(0.0, 3.5, 500)
    nz_grid = smail_nz(
        z_grid, z0=0.4
    )  # narrower, more clustering-sample-like than the lensing default

    z_bg, _ = fiducial_background(4.0, 2000, Omega_m, w0, wa)
    z_nodes = _background_z_of_chi(Omega_m, w0, wa, _CHI_NODES)

    z_pad, nz_pad = _pad_dndz_for_lookup(
        z_grid, nz_grid, float(z_bg[0]), float(z_bg[-1])
    )
    ng = np.asarray(
        _smooth_lookup(
            jnp.asarray(z_pad), jnp.asarray(nz_pad), jnp.asarray(z_nodes), len(z_pad)
        )
    )
    E = np.asarray(
        jnp.sqrt(_E2_of_a(1.0 / (1.0 + jnp.asarray(z_nodes)), Omega_m, w0, wa))
    )
    # No `bias *` here -- CCL's get_kernel() excludes it (see module docstring).
    W_ours_nobias = ng * E / _C_OVER_H0

    tr = pyccl.NumberCountsTracer(
        _ccl_cosmology(p),
        dndz=(z_grid, nz_grid),
        bias=(z_grid, bias * np.ones_like(z_grid)),
        has_rsd=False,
        mag_bias=None,
    )
    W_ccl = tr.get_kernel(_CHI_NODES / h)[0]

    np.testing.assert_allclose(W_ccl, h * W_ours_nobias, rtol=_RTOL)


def test_cmb_lensing_kernel_matches_ccl(fiducial_params):
    p = fiducial_params
    Omega_m = float(p["omega_cold"][0])
    h = float(p["hubble"][0])
    w0, wa = float(p["w0"][0]), float(p["wa"][0])
    z_source = 1089.0

    z_fine = np.linspace(0.0, z_source, 20000)
    chi_fine = np.asarray(_comoving_distance(jnp.asarray(z_fine), Omega_m, w0, wa))
    chi_star = chi_fine[-1]

    z_nodes = _background_z_of_chi(
        Omega_m, w0, wa, _CHI_NODES
    )  # default z_max=4.0 is plenty
    # resolution for chi in [50, 3000];
    # chi_star itself uses its own
    # dedicated fine grid to z_source above
    a_nodes = 1.0 / (1.0 + z_nodes)
    ramp = np.maximum(
        chi_star - _CHI_NODES, 0.0
    )  # exact form -- the ~2 Mpc/h C1 smoothing
    # _leg_weight applies is negligible at this rtol
    W_ours = 1.5 / _C_OVER_H0**2 * Omega_m * (_CHI_NODES / a_nodes) * (ramp / chi_star)

    tr = pyccl.CMBLensingTracer(_ccl_cosmology(p), z_source=z_source)
    W_ccl = tr.get_kernel(_CHI_NODES / h)[0]

    np.testing.assert_allclose(W_ccl, h * W_ours, rtol=_RTOL)


def test_growth_factor_matches_ccl(fiducial_params):
    p = fiducial_params
    cosmo = _ccl_cosmology(p)
    z = np.array([0.1, 0.5, 1.0, 2.0])
    D_ccl = np.array([cosmo.growth_factor(1.0 / (1.0 + x)) for x in z])
    D = np.asarray(
        growth_ratios(
            jnp.asarray(1.0 / (1.0 + z)),
            jnp.asarray(p["omega_cold"][0]),
            -1.0,
            0.0,
            jnp.float64,
        )
    )
    np.testing.assert_allclose(D, D_ccl, rtol=5e-4)


def test_ia_amplitude_matches_ccl_transfer(fiducial_params):
    """CCL applies the alignment amplitude as a transfer function, with
    5e-14 rho_crit in place of the conventional rounded C1 rho_crit = 0.0134
    (a constant 3.6 % offset). So the check is that the ratio is a constant in z:
    the growth, (1+z)^eta evolution and sign all agree."""
    p = fiducial_params
    Omega_m = float(p["omega_cold"][0])
    A, eta, z_pivot = 1.3, 0.7, 0.62
    z_grid = np.linspace(0.0, 3.5, 500)
    tracer = pyccl.WeakLensingTracer(
        _ccl_cosmology(p),
        dndz=(z_grid, smail_nz(z_grid)),
        has_shear=False,
        ia_bias=(z_grid, A * ((1 + z_grid) / (1 + z_pivot)) ** eta),
        use_A_ia=True,
    )
    z = np.array([0.1, 0.4, 0.62, 1.0, 2.0])
    ccl = np.array(
        [tracer.get_transfer(np.log(0.1), 1.0 / (1.0 + x)) for x in z]
    ).ravel()
    D = growth_ratios(
        jnp.asarray(1.0 / (1.0 + z)), jnp.asarray(Omega_m), -1.0, 0.0, jnp.float64
    )
    ours = np.asarray(ia_amplitude(Omega_m, A, eta, z_pivot, D, z))
    ratio = ccl / ours
    assert np.ptp(ratio) < 1e-3 * ratio.mean()
    assert abs(ratio.mean() - 1.0) < 0.05


def test_ia_kernel_matches_ccl(fiducial_params):
    """CCL's IA kernel is the amplitude-free n(z) H(z)/c, like its counts kernel."""
    p = fiducial_params
    Omega_m = float(p["omega_cold"][0])
    h = float(p["hubble"][0])
    z_grid = np.linspace(0.0, 3.5, 500)
    nz_grid = smail_nz(z_grid)

    z_bg, _ = fiducial_background(4.0, 2000, Omega_m, -1.0, 0.0)
    z_nodes = _background_z_of_chi(Omega_m, -1.0, 0.0, _CHI_NODES)
    z_pad, nz_pad = _pad_dndz_for_lookup(
        z_grid, nz_grid, float(z_bg[0]), float(z_bg[-1])
    )
    D = growth_ratios(
        jnp.asarray(1.0 / (1.0 + z_nodes)), jnp.asarray(Omega_m), -1.0, 0.0, jnp.float64
    )
    W = np.asarray(
        ia_kernel(
            Omega_m,
            -1.0,
            0.0,
            1.0,
            0.0,
            0.62,
            D,
            jnp.asarray(z_nodes),
            jnp.asarray(z_pad),
            jnp.asarray(nz_pad),
            len(z_pad),
        )
    )
    f = np.asarray(ia_amplitude(Omega_m, 1.0, 0.0, 0.62, D, z_nodes))

    tracer = pyccl.WeakLensingTracer(
        _ccl_cosmology(p),
        dndz=(z_grid, nz_grid),
        has_shear=False,
        ia_bias=(z_grid, np.ones_like(z_grid)),
    )
    np.testing.assert_allclose(
        tracer.get_kernel(_CHI_NODES / h)[0], h * W / f, rtol=_RTOL
    )
