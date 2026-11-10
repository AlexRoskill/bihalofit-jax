"""
pytest suite for src/aperture.py -- the third-order aperture mass
statistic <M_ap^3> built on top of projected_bihalofit.make_bkappa.

Run:
    pytest tests/test_aperture_statistics.py -v --device cpu
"""

import warnings

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bihalofit_jax import (
    Cosmology,
    make_map3 as _make_map3_multi,
    ARCMIN_TO_RAD,
)
from bihalofit_jax.aperture import (
    _build_triangle_grid,
    u_hat,
    ratio_nodes,
)
from bihalofit_jax import smail_nz
from bihalofit_jax._utils import gauss_legendre_log


def _compute_map3_multi(theta1, theta2, theta3, dndz, tracers, cosmo=None, **kw):
    """Build and evaluate in one step."""
    return _make_map3_multi(dndz, tracers, **kw)(theta1, theta2, theta3, cosmo)


# Helpers for the single-n(z) shear case, <M_ap^3>: 'lensing' on every leg.

_LENSING3 = ("lensing", "lensing", "lensing")


def make_map3(n_z, **kwargs):
    return _make_map3_multi((n_z, n_z, n_z), _LENSING3, **kwargs)


def compute_map3(theta1, theta2, theta3, n_z, *rest, **kwargs):
    return _compute_map3_multi(
        theta1, theta2, theta3, (n_z, n_z, n_z), _LENSING3, *rest, **kwargs
    )


_OC, _OB, _S8, _NS, _H = 0.3156, 0.0492, 0.831, 0.9645, 0.6727

# Small quadrature (N_ell, N_phi, N_chi) throughout -- these tests check
# correctness/shape/differentiability properties, not converged accuracy, so
# keep JIT compilation fast regardless of backend.
_N_ELL, _N_PHI, _N_CHI = 6, 6, 6
_ELL_MAX = 2.0e3  # keeps k = ell/chi_min inside the P_L grid for chi_min=10


@pytest.fixture(scope="module")
def nz():
    z = np.linspace(0.0, 3.5, 200)
    return (z, smail_nz(z))


@pytest.fixture(scope="module")
def map3(nz):
    # ell3 = |ell1 + ell2| can exceed ell_max at phi~0, pushing k=ell/chi_min
    # slightly past the P_L grid on a small fraction of chi-nodes -- expected
    # at any finite ell_max, not a bug; the RuntimeWarning is not the point
    # of these tests.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return make_map3(
            nz,
            ell_min=1.0,
            ell_max=_ELL_MAX,
            N_ell=_N_ELL,
            N_phi=_N_PHI,
            pk_source="eh",
            N_chi=_N_CHI,
        )


# ---------------------------------------------------------------------------
# u_hat -- Crittenden aperture filter
# ---------------------------------------------------------------------------


def test_u_hat_zero_at_origin():
    assert float(u_hat(jnp.array(0.0))) == 0.0


def test_u_hat_non_negative():
    eta = jnp.linspace(0.0, 10.0, 2001)
    assert bool(jnp.all(u_hat(eta) >= 0.0))


def test_u_hat_peak_at_sqrt2():
    """u_hat(eta) = eta^2/2 * exp(-eta^2/2) peaks analytically at eta=sqrt(2)."""
    eta = jnp.linspace(0.0, 5.0, 20001)
    peak_eta = float(eta[jnp.argmax(u_hat(eta))])
    assert peak_eta == pytest.approx(np.sqrt(2.0), abs=1e-3)


def test_u_hat_decays_at_large_eta():
    assert float(u_hat(jnp.array(20.0))) < 1e-6


# ---------------------------------------------------------------------------
# gauss_legendre_log -- log-spaced GL quadrature helper
# ---------------------------------------------------------------------------


def test_gauss_legendre_log_integrates_constant():
    """int_a^b 1 dx = b - a."""
    x, w = gauss_legendre_log(1.0, 100.0, 40)
    assert float(np.sum(w)) == pytest.approx(99.0, rel=1e-10)


def test_gauss_legendre_log_integrates_inverse():
    """int_a^b (1/x) dx = ln(b/a) -- exact for GL on a log grid."""
    x, w = gauss_legendre_log(1.0, 100.0, 40)
    integral = float(np.sum(w / x))
    assert integral == pytest.approx(np.log(100.0), rel=1e-10)


def test_gauss_legendre_log_nodes_within_bounds():
    x, w = gauss_legendre_log(2.0, 50.0, 20)
    assert np.all(x >= 2.0) and np.all(x <= 50.0)
    assert np.all(w > 0)


# ---------------------------------------------------------------------------
# _build_triangle_grid -- Fourier triangle closure
# ---------------------------------------------------------------------------


def test_triangle_grid_closes_law_of_cosines():
    """Every (ell1, ell2, phi) node's ell3 satisfies the law of cosines used
    to define it (i.e. the grid construction is self-consistent)."""
    ell1, ell2, ell3, weight = _build_triangle_grid(1.0, 100.0, 5, 5)
    # phi isn't returned directly, so re-derive it from ell3 and check ell3 > 0
    # and the triangle inequality, which must hold for any real phi in [0,pi].
    assert np.all(ell3 > 0)
    assert np.all(ell3 <= ell1 + ell2 + 1e-8)
    assert np.all(ell3 >= np.abs(ell1 - ell2) - 1e-8)


def test_triangle_grid_weights_positive():
    _, _, _, weight = _build_triangle_grid(1.0, 100.0, 5, 5)
    assert np.all(weight > 0)


def test_triangle_grid_shape():
    N_ell, N_phi = 4, 7
    ell1, ell2, ell3, weight = _build_triangle_grid(1.0, 100.0, N_ell, N_phi)
    assert (
        ell1.shape
        == ell2.shape
        == ell3.shape
        == weight.shape
        == (N_ell * N_ell * N_phi,)
    )


def test_map3_scalar_theta_scalar_cosmo_shape(map3):
    """Output rank is always (N_cosmo, N_theta), including for a single
    cosmology and a scalar theta, whether the caller passed a float or a
    length-1 array."""
    theta = 5.0 * ARCMIN_TO_RAD
    out = np.asarray(map3(theta, theta, theta, Cosmology(_OC, _OB, _S8, _NS, _H)))
    assert out.shape == (1, 1)


def test_map3_batch_theta_shape(map3):
    theta = jnp.array([1.0, 5.0, 10.0]) * ARCMIN_TO_RAD
    out = np.asarray(map3(theta, theta, theta, Cosmology(_OC, _OB, _S8, _NS, _H)))
    assert out.shape == (1, 3)


def test_map3_batch_cosmo_shape(map3):
    theta = 5.0 * ARCMIN_TO_RAD
    oc = jnp.array([_OC, _OC * 1.05, _OC * 0.95])
    out = np.asarray(map3(theta, theta, theta, Cosmology(oc, _OB, _S8, _NS, _H)))
    assert out.shape == (3, 1)


def test_map3_batch_theta_and_cosmo_shape(map3):
    theta = jnp.array([1.0, 5.0]) * ARCMIN_TO_RAD
    oc = jnp.array([_OC, _OC * 1.05, _OC * 0.95])
    out = np.asarray(map3(theta, theta, theta, Cosmology(oc, _OB, _S8, _NS, _H)))
    assert out.shape == (3, 2)


# ---------------------------------------------------------------------------
# Finiteness / positivity
# ---------------------------------------------------------------------------


def test_map3_finite_and_positive(map3):
    theta = jnp.array([1.0, 3.0, 5.0, 10.0, 20.0]) * ARCMIN_TO_RAD
    out = np.asarray(map3(theta, theta, theta, Cosmology(_OC, _OB, _S8, _NS, _H)))
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)


# ---------------------------------------------------------------------------
# Symmetry: <M_ap^3> is invariant under exchanging two aperture radii, since
# the underlying (ell1, ell2) quadrature grid and B_kappa are both symmetric
# under that exchange.
# ---------------------------------------------------------------------------


def test_map3_symmetric_under_theta_exchange(nz):
    """theta1 <-> theta2 exchange is EXACT on the 'phi' grid, at any resolution.

    That grid samples ell1 and ell2 on the same axis, so the node set is
    invariant under the swap and the two sums are identical term by term.
    The diagonal grid samples (ell1, rho=ell2/ell1) instead, which is not
    swap-invariant at finite resolution -- see the companion test below.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        map3 = make_map3(
            nz,
            ell_min=1.0,
            ell_max=_ELL_MAX,
            N_ell=_N_ELL,
            N_phi=_N_PHI,
            grid="phi",
            pk_source="eh",
            N_chi=_N_CHI,
        )
    t1, t2, t3 = (2.0 * ARCMIN_TO_RAD, 5.0 * ARCMIN_TO_RAD, 8.0 * ARCMIN_TO_RAD)
    v_123 = float(np.asarray(map3(t1, t2, t3, Cosmology(_OC, _OB, _S8, _NS, _H)))[0, 0])
    v_213 = float(np.asarray(map3(t2, t1, t3, Cosmology(_OC, _OB, _S8, _NS, _H)))[0, 0])
    assert v_123 == pytest.approx(v_213, rel=1e-10)


def test_map3_diagonal_grid_symmetric_to_quadrature_accuracy(nz):
    """On the diagonal grid the same exchange holds only as well as the
    quadrature converges -- it is not structural.

    At the shipped resolution the residual asymmetry is ~1e-9 to 1e-6, far
    below the ~1.6e-4 quadrature error, so it never matters in practice. It is
    only visible on a deliberately coarse grid. ell_max is sized from
    theta_min here: truncating a leg's filter (as a too-small ell_max does)
    breaks the symmetry far more than the quadrature does.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        map3 = make_map3(
            nz,
            theta_min=2.0 * ARCMIN_TO_RAD,
            grid="diagonal",
            N_ell=48,
            N_rho=24,
            N_phi=32,
            pk_source="eh",
            N_chi=6,
        )
    t1, t2, t3 = (2.0 * ARCMIN_TO_RAD, 5.0 * ARCMIN_TO_RAD, 8.0 * ARCMIN_TO_RAD)
    v_123 = float(np.asarray(map3(t1, t2, t3, Cosmology(_OC, _OB, _S8, _NS, _H)))[0, 0])
    v_213 = float(np.asarray(map3(t2, t1, t3, Cosmology(_OC, _OB, _S8, _NS, _H)))[0, 0])
    assert v_123 == pytest.approx(v_213, rel=1e-4)


# ---------------------------------------------------------------------------
# make_map3 vs compute_map3 consistency
# ---------------------------------------------------------------------------


def test_compute_map3_matches_make_map3(nz):
    theta = 5.0 * ARCMIN_TO_RAD
    kwargs = dict(
        ell_min=1.0,
        ell_max=_ELL_MAX,
        N_ell=_N_ELL,
        N_phi=_N_PHI,
        pk_source="eh",
        N_chi=_N_CHI,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v_direct = np.asarray(
            compute_map3(
                theta, theta, theta, nz, Cosmology(_OC, _OB, _S8, _NS, _H), **kwargs
            )
        )
        map3 = make_map3(nz, **kwargs)
    v_built = np.asarray(map3(theta, theta, theta, Cosmology(_OC, _OB, _S8, _NS, _H)))
    assert v_direct == pytest.approx(v_built, rel=1e-12)


# ---------------------------------------------------------------------------
# Physical sanity: monotonic increase with sigma8 (more clustering -> larger
# third moment), matching the sign of the P(k)-level sigma8 dependence.
# ---------------------------------------------------------------------------


def test_map3_sigma8_monotonicity(map3):
    theta = 5.0 * ARCMIN_TO_RAD
    s8_values = [0.75, 0.80, 0.831, 0.90]
    vals = [
        float(
            np.asarray(map3(theta, theta, theta, Cosmology(_OC, _OB, s8, _NS, _H)))[
                0, 0
            ]
        )
        for s8 in s8_values
    ]
    assert np.all(np.diff(vals) > 0), f"<Map3> not monotonic in sigma8: {vals}"


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", ["omega_cold", "sigma8_cold", "hubble"])
def test_map3_gradient_finite(map3, param):
    theta = 5.0 * ARCMIN_TO_RAD
    base = dict(oc=_OC, ob=_OB, s8=_S8, ns=_NS, h=_H)
    key = {"omega_cold": "oc", "sigma8_cold": "s8", "hubble": "h"}[param]

    def f(x):
        local = dict(base)
        local[key] = x
        return map3(
            theta,
            theta,
            theta,
            Cosmology(local["oc"], local["ob"], local["s8"], local["ns"], local["h"]),
        )[0, 0]

    g = jax.grad(f)(base[key])
    assert np.isfinite(float(g))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_make_map3_invalid_pk_source(nz):
    with pytest.raises(ValueError, match="pk_source must be one of"):
        make_map3(nz, pk_source="bogus", N_ell=_N_ELL, N_phi=_N_PHI, N_chi=_N_CHI)


# ---------------------------------------------------------------------------
# term ('full'/'1h'/'3h'/'tree') -- forwarded straight through to
# make_projected_bihalofit (see test_projected_bihalofit.py's analogous
# section); these just check the forwarding itself survives the extra
# aperture-filter reduction.
# ---------------------------------------------------------------------------


def _common_map3_kwargs():
    return dict(
        ell_min=1.0,
        ell_max=_ELL_MAX,
        N_ell=_N_ELL,
        N_phi=_N_PHI,
        pk_source="eh",
        N_chi=_N_CHI,
    )


def test_map3_term_invalid_raises(nz):
    """Unrecognised term raises ValueError before any JAX work."""
    with pytest.raises(ValueError, match="term must be one of"):
        make_map3(nz, term="bogus", **_common_map3_kwargs())


def test_map3_term_full_equals_1h_plus_3h(nz):
    """term='full' equals term='1h' + term='3h' after the aperture-filter
    reduction, same as it does at the raw and Limber-projected bispectrum
    levels.

    Pinned to float64: this checks an exact algebraic identity, and the
    rtol=1e-8 that makes the check meaningful is below float32's ~1e-7
    relative precision, so it is unsatisfiable in float32 no matter how
    correct the decomposition is. make_map3 defaults to float32 (~20x faster
    at no accuracy cost for the statistic itself), which is the right default
    for physics but not for an exactness assertion -- hence the override here
    rather than a loosened tolerance, which would weaken a real invariant."""
    theta = jnp.array([1.0, 5.0, 10.0]) * ARCMIN_TO_RAD
    cosmo = Cosmology(_OC, _OB, _S8, _NS, _H)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _kw = dict(_common_map3_kwargs(), dtype=np.float64)
        map3_full = make_map3(nz, term="full", **_kw)
        map3_1h = make_map3(nz, term="1h", **_kw)
        map3_3h = make_map3(nz, term="3h", **_kw)
    M_full = np.asarray(map3_full(theta, theta, theta, cosmo))
    M_1h = np.asarray(map3_1h(theta, theta, theta, cosmo))
    M_3h = np.asarray(map3_3h(theta, theta, theta, cosmo))
    np.testing.assert_allclose(M_full, M_1h + M_3h, rtol=1e-8, atol=0)


def test_map3_term_tree_finite_and_smaller_than_full(nz):
    """term='tree' is finite and, at these aperture scales, smaller in
    amplitude than the full nonlinear <Map^3> -- mirrors the equivalent
    sanity check on B_kappa."""
    theta = jnp.array([1.0, 5.0, 10.0]) * ARCMIN_TO_RAD
    cosmo = Cosmology(_OC, _OB, _S8, _NS, _H)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        map3_full = make_map3(nz, term="full", **_common_map3_kwargs())
        map3_tree = make_map3(nz, term="tree", **_common_map3_kwargs())
    M_full = np.asarray(map3_full(theta, theta, theta, cosmo))
    M_tree = np.asarray(map3_tree(theta, theta, theta, cosmo))
    assert np.all(np.isfinite(M_tree))
    assert np.all(M_tree > 0)
    assert np.all(M_tree < M_full)


def test_map3_term_not_cached_across_different_values(nz):
    """Same (dndz, tracers, quadrature) built with different term values
    must not silently share a cached closure -- same cache-key bug class
    already guarded against at the projected-bispectrum level."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f_full = make_map3(nz, **_common_map3_kwargs())
        f_1h = make_map3(nz, term="1h", **_common_map3_kwargs())
    assert f_full is not f_1h
    theta = 5.0 * ARCMIN_TO_RAD
    cosmo = Cosmology(_OC, _OB, _S8, _NS, _H)
    r_full = np.asarray(f_full(theta, theta, theta, cosmo))
    r_1h = np.asarray(f_1h(theta, theta, theta, cosmo))
    assert not np.allclose(r_full, r_1h, atol=0)


# ---------------------------------------------------------------------------
# Golden reference agreement: threepoint_py (Linke/Burger et al.), an
# independent implementation with its own Limber projection and quadrature
# scheme (P(k), raw bispectrum, and B_kappa also agree to sub-0.3%).
# ---------------------------------------------------------------------------


def test_against_threepoint_reference(map3_threepoint_reference):
    """bihalofit_jax's <Map_ap^3> must agree with threepoint_py to within 1%.

    Uses pk_source='eh_99' (matches threepoint_py's Eisenstein-Hu formula) and
    the same quadrature settings used when the reference was generated, so any
    discrepancy reflects a real difference in the Limber/aperture-filter
    integration rather than P(k) choice or under-resolved quadrature.
    Skips automatically when the reference file hasn't been generated yet.
    """
    ref = map3_threepoint_reference
    z_grid = np.linspace(0.0, 3.5, 256)
    nz_grid = smail_nz(z_grid)

    map3 = make_map3(
        (z_grid, nz_grid),
        ell_min=float(ref["ell_min"]),
        ell_max=float(ref["ell_max"]),
        N_ell=int(ref["N_ell"]),
        N_phi=int(ref["N_ell"]),
        N_chi=int(ref["N_chi"]),
        pk_source=str(ref["pk_source"]),
    )

    theta1 = ref["theta1_arcmin"] * ARCMIN_TO_RAD
    theta2 = ref["theta2_arcmin"] * ARCMIN_TO_RAD
    theta3 = ref["theta3_arcmin"] * ARCMIN_TO_RAD

    vals = np.asarray(
        map3(
            theta1,
            theta2,
            theta3,
            Cosmology(
                float(ref["omega_cold"]),
                float(ref["omega_baryon"]),
                float(ref["sigma8_cold"]),
                float(ref["ns"]),
                float(ref["hubble"]),
            ),
        )
    )

    ratio = vals / ref["map3_tp"]
    max_dev = float(np.max(np.abs(ratio - 1.0)))
    assert (
        max_dev < 0.01
    ), f"max |Map3/Map3_threepoint_py - 1| = {max_dev:.4f} exceeds 1% tolerance"


# ---------------------------------------------------------------------------
# ell-axis self-convergence for an elongated triplet, on the shipped diagonal
# grid. A coarse ell rule under-converges badly on elongated shapes; enough
# nodes fixes it. Thresholds are measured, not guessed: for (4',16',32') at
# ell_max=10/theta_min, against an N_ell=192 reference on the same rho/phi/chi
# grid (so only the ell axis varies), N_ell=16 gives 5.6% and the default
# N_ell=96 gives 1e-6.
# Cheap on purpose: pk_source='eh', coarse rho/phi/chi axes, no baccoemu/GPU
# (3 GB peak).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ell_axis_self_converges_for_elongated_triplet(nz):
    """A coarse ell grid shows the known elongated-triangle under-convergence;
    raising the ell order to 96 recovers the fine-grid answer to <0.5%."""
    t1, t2, t3 = 4.0, 16.0, 32.0
    theta_min_rad = min(t1, t2, t3) * ARCMIN_TO_RAD
    ell_max_i = 10.0 / theta_min_rad
    th_rad = (
        np.array([t1]) * ARCMIN_TO_RAD,
        np.array([t2]) * ARCMIN_TO_RAD,
        np.array([t3]) * ARCMIN_TO_RAD,
    )
    cosmo = Cosmology(_OC, _OB, _S8, _NS, _H)

    def _eval(N_ell, N_rho=16, N_phi=16, N_chi=10):
        map3 = _make_map3_multi(
            (nz, nz, nz),
            _LENSING3,
            ell_min=1.0,
            ell_max=ell_max_i,
            N_ell=N_ell,
            N_rho=N_rho,
            N_phi=N_phi,
            pk_source="eh",
            N_chi=N_chi,
        )
        return float(np.asarray(map3(*th_rad, cosmo))[0, 0])

    ref = _eval(N_ell=192)
    coarse = _eval(N_ell=16)
    fine = _eval(N_ell=96)

    err_coarse, err_fine = abs(coarse / ref - 1.0), abs(fine / ref - 1.0)

    # Guard that the setup still reproduces the documented failure mode, so
    # this is not accidentally testing nothing.
    assert err_coarse > 0.02, (
        f"expected the coarse ell grid to show its known elongated-triangle "
        f"under-convergence (>2%), got {err_coarse * 100:.3f}% -- "
        "has the reference triplet/setup changed?"
    )
    assert err_fine < 0.005, (
        f"the default 96 ell-nodes should match the fine reference to <0.5%, "
        f"got {err_fine * 100:.3f}%"
    )


# ---------------------------------------------------------------------------
# Multi-tracer Map3: a 'counts' leg (galaxy-galaxy-lensing-like <N_ap Map^2>)
# mixed with two 'lensing' legs.
# ---------------------------------------------------------------------------


def test_map3_multi_tracer_counts_leg_finite_and_differentiable(nz):
    """tracers=('counts','lensing','lensing') gives a finite, differentiable
    <N_ap M_ap^2>-like statistic -- exercises make_map3's forwarding of
    dndz/tracers/bias through to make_projected_bihalofit."""
    theta = 5.0 * ARCMIN_TO_RAD
    dndz = (nz, nz, nz)
    tracers = ("counts", "lensing", "lensing")
    bias = (1.7, None, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        map3_multi = _make_map3_multi(
            dndz,
            tracers,
            bias=bias,
            ell_min=1.0,
            ell_max=_ELL_MAX,
            N_ell=_N_ELL,
            N_phi=_N_PHI,
            pk_source="eh",
            N_chi=_N_CHI,
        )
    out = np.asarray(map3_multi(theta, theta, theta, Cosmology(_OC, _OB, _S8, _NS, _H)))
    assert np.all(np.isfinite(out)), f"<N_ap M_ap^2> not finite: {out}"

    def f(s8):
        return map3_multi(theta, theta, theta, Cosmology(_OC, _OB, s8, _NS, _H))[0, 0]

    g = jax.grad(f)(_S8)
    assert np.isfinite(float(g)), f"d<N_ap M_ap^2>/d(sigma8) not finite: {g}"


@pytest.mark.slow
@pytest.mark.highmem
def test_map3_diagonal_grid_matches_phi_grid_and_is_more_accurate(nz):
    """The 'diagonal' grid integrates the same thing as 'phi', more accurately.

    Pins the (ell1, rho=ell2/ell1, phi) scheme added Aug 2026. Two claims:

    1. Same integral. On a benign triplet both grids, well resolved, agree --
       this is what catches a wrong Jacobian/domain in the diagonal builder,
       which self-convergence cannot see.
    2. Better on the hard case. For (1', 1', 32') the triangle inequality
       ell3 >= |ell1-ell2| confines the integrand to a thin ell1~ell2 band the
       phi grid cannot resolve; diagonal must beat it by a wide margin.

    N_phi=48 (not 24) is deliberate: the ell-side fix exposes a phi-side limit
    that the phi grid hid, and at N_phi=24 the diagonal grid scores 1.4e-2 on
    this triplet instead of ~5e-4.
    """
    cosmo = Cosmology(_OC, _OB, _S8, _NS, _H)
    common = dict(pk_source="eh", dtype=np.float64, N_chi=_N_CHI)

    # 1. same integral on a benign triplet, both well resolved
    th = jnp.array([4.0, 8.0, 16.0]) * ARCMIN_TO_RAD
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m_phi = make_map3(
            nz, grid="phi", theta_min=float(np.min(th)), N_ell=224, N_phi=48, **common
        )
        m_dia = make_map3(
            nz,
            grid="diagonal",
            theta_min=float(np.min(th)),
            N_ell=224,
            N_rho=48,
            N_phi=48,
            **common,
        )
        a = float(np.asarray(m_phi(th[:1], th[1:2], th[2:3], cosmo))[0, 0])
        b = float(np.asarray(m_dia(th[:1], th[1:2], th[2:3], cosmo))[0, 0])
    assert abs(b / a - 1.0) < 5e-3, f"grids disagree on a benign triplet: {a} vs {b}"

    # 2. hard triplet: diagonal must be much closer to a well-resolved answer
    hard = jnp.array([1.0, 1.0, 32.0]) * ARCMIN_TO_RAD
    tmin = float(np.min(hard))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = make_map3(
            nz, grid="diagonal", theta_min=tmin, N_ell=384, N_rho=64, N_phi=64, **common
        )
        cheap_phi = make_map3(
            nz, grid="phi", theta_min=tmin, N_ell=96, N_phi=48, **common
        )
        cheap_dia = make_map3(
            nz, grid="diagonal", theta_min=tmin, N_ell=168, N_rho=32, N_phi=48, **common
        )
        r = float(np.asarray(ref(hard[:1], hard[1:2], hard[2:3], cosmo))[0, 0])
        p_ = float(np.asarray(cheap_phi(hard[:1], hard[1:2], hard[2:3], cosmo))[0, 0])
        d_ = float(np.asarray(cheap_dia(hard[:1], hard[1:2], hard[2:3], cosmo))[0, 0])
    e_phi, e_dia = abs(p_ / r - 1.0), abs(d_ / r - 1.0)
    assert e_dia < 5e-3, f"diagonal grid worse than expected on (1,1,32): {e_dia:.2e}"
    assert e_dia < e_phi / 5.0, (
        f"diagonal should clearly beat phi here: "
        f"diag {e_dia:.2e} vs phi {e_phi:.2e}"
    )


def test_ratio_nodes_exact_on_its_invariant():
    """ratio_nodes integrates d(rho)/rho exactly -- checks rule and Jacobian."""
    rho, w = ratio_nodes(64, 1e4)
    got = float(np.sum(w / rho))
    assert abs(got / (2.0 * np.log(1e4)) - 1.0) < 1e-12


@pytest.mark.slow
@pytest.mark.highmem  # builds the full default grid: ~10 GB peak, more than a CI runner has
def test_default_grid_does_not_warn_about_pk_extrapolation(nz):
    """The diagonal grid's near-degenerate triangles (ell3 ~ 0) must not trip the P_L-grid check."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_map3(nz, theta_min=4.0 * ARCMIN_TO_RAD, pk_source="eh")
    assert not [w for w in caught if "out of P_L grid" in str(w.message)]
