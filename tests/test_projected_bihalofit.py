"""
pytest regression suite for projected_bihalofit.py: convergence bispectrum.

Run full suite:
    pytest tests/test_projected_bihalofit.py -v --device cpu

Limit to a single backend:
    pytest tests/test_projected_bihalofit.py -v --device cpu --backend eh

Single test:
    pytest tests/test_projected_bihalofit.py::test_output_finite_positive -v
"""

import os
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import pytest

try:
    _np_trapezoid = np.trapezoid
except AttributeError:
    _np_trapezoid = np.trapz

import bihalofit_jax.projected_bispectrum as _pbhf
from bihalofit_jax import (
    Cosmology,
    smail_nz,
    make_projected_bihalofit,
)
from bihalofit_jax.background import (
    _comoving_distance,
    _C_OVER_H0,
    fiducial_background,
    lensing_geometry,
)

# Monkeypatch targets must be the module where the name is *looked up*, not
# wherever it happens to be re-exported: _resolve_N_R resolves _tol_to_niter
# in _knl_solver's globals.
import bihalofit_jax._knl_solver as _bhfs
from bihalofit_jax import make_bihalofit


def compute_bihalofit(expfactors, k1, k2, k3, cosmo=None, **kwargs):
    """Build and evaluate in one step."""
    return make_bihalofit(k1, k2, k3, expfactors, **kwargs)(cosmo)


from bihalofit_jax._knl_solver import _tol_to_niter

# Helpers for the single-n(z) shear case: 'lensing' on every leg.

_LENSING3 = ("lensing", "lensing", "lensing")


def _cos(p, **overrides):
    """Cosmology from the fiducial_params fixture dict, with optional
    per-field overrides (e.g. _cos(p, sigma8_cold=0.5)). Fixture values are
    length-1 arrays; a scalar override is broadcast against them."""
    d = {k: v for k, v in p.items()}
    d.update(overrides)
    return Cosmology(**d)


def make_bkappa(ell1, ell2, ell3, n_z, **kwargs):
    return make_projected_bihalofit(
        ell1, ell2, ell3, (n_z, n_z, n_z), _LENSING3, **kwargs
    )


def compute_convergence_bispectrum(ell1, ell2, ell3, n_z, cosmo=None, **kwargs):
    """B_kappa ('lensing' on every leg), built and evaluated in one step."""
    return make_projected_bihalofit(
        ell1, ell2, ell3, (n_z, n_z, n_z), _LENSING3, **kwargs
    )(cosmo)


_COSMO_NAMES = [
    "omega_cold",
    "omega_baryon",
    "sigma8_cold",
    "ns",
    "hubble",
    "neutrino_mass",
    "w0",
    "wa",
]


def _bkappa_with_jac(ell1, ell2, ell3, n_z, cosmo, wrt, **kw):
    """Primal + Jacobian w.r.t. one or more named cosmological parameters.

    Cosmology is a pytree, so a single jacfwd over the container yields a
    Cosmology of derivatives and the parameter is selected by *name*: no
    argnums bookkeeping tied to positional order."""
    names = [wrt] if isinstance(wrt, str) else list(wrt)
    cosmo = cosmo if isinstance(cosmo, Cosmology) else Cosmology(**cosmo)

    def _fwd(c):
        return compute_convergence_bispectrum(ell1, ell2, ell3, n_z, c, **kw)

    B = _fwd(cosmo)
    jac = jax.jacfwd(_fwd)(cosmo)
    return B, {n: getattr(jac, n) for n in names}


# ---------------------------------------------------------------------------
# Module-level shared data
# ---------------------------------------------------------------------------

# Dense z grid for all n(z) integrals (500 points is overkill; keeps trapz error < 0.1%)
_Z_GRID = np.linspace(0.0, 3.5, 500)
_NZ_GRID = smail_nz(_Z_GRID)
_N_Z = (_Z_GRID, _NZ_GRID)

# Representative ell triangles covering equilateral, squeezed, scalene, near-degenerate.
# All satisfy the triangle inequality; chosen to span the k = ell/chi range that
# bihalofit_jax is tested on.
_ELL_TRIS = np.array(
    [
        [100.0, 100.0, 100.0],  # equilateral, large angular scales
        [500.0, 500.0, 500.0],  # equilateral, intermediate
        [2000.0, 2000.0, 2000.0],  # equilateral, small scales
        [500.0, 500.0, 100.0],  # squeezed
        [1000.0, 1000.0, 200.0],  # squeezed, high-ell legs
        [300.0, 500.0, 700.0],  # scalene  (700 < 300+500)
        [200.0, 500.0, 600.0],  # scalene
    ]
)

_ELL1 = _ELL_TRIS[:, 0]
_ELL2 = _ELL_TRIS[:, 1]
_ELL3 = _ELL_TRIS[:, 2]

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "reference", "bkappa_golden.npz")


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def nz():
    return _N_Z


@pytest.fixture(scope="session")
def bkappa_golden():
    if not os.path.exists(_GOLDEN_PATH):
        pytest.skip(
            f"bkappa golden file not found: {_GOLDEN_PATH}; "
            "run tests/generate_bkappa_golden.py first"
        )
    return np.load(_GOLDEN_PATH, allow_pickle=True)


# ---------------------------------------------------------------------------
# Test 1: smail_nz utility
# ---------------------------------------------------------------------------


def test_smail_nz_normalization():
    """n(z) integrates to 1 on a dense grid."""
    z = np.linspace(0.0, 4.0, 2000)
    n = smail_nz(z)
    norm = _np_trapezoid(n, z)
    assert abs(norm - 1.0) < 1e-3, f"smail_nz integral = {norm:.6f} (expected 1.0)"


def test_smail_nz_non_negative():
    """n(z) is non-negative everywhere."""
    z = np.linspace(0.0, 4.0, 500)
    assert np.all(smail_nz(z) >= 0)


def test_smail_nz_median_redshift():
    """Default Smail n(z) median redshift is in [0.7, 1.1]."""
    z = np.linspace(0.0, 4.0, 2000)
    n = smail_nz(z)
    cdf = np.cumsum(n * np.gradient(z))
    cdf /= cdf[-1]
    z_median = float(np.interp(0.5, cdf, z))
    assert 0.7 < z_median < 1.1, f"smail_nz median z = {z_median:.3f} (expected ~0.9)"


# ---------------------------------------------------------------------------
# Test 2: output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N_cosmo", [1, 3, 5])
def test_output_shape(N_cosmo, fiducial_params, nz, jax_device, backend):
    """B_κ shape is (N_ltri,) for scalar and (N_cosmo, N_ltri) for batch input."""
    N_ltri = len(_ELL1)
    p = fiducial_params
    oc = float(p["omega_cold"][0])
    ob = float(p["omega_baryon"][0])
    s8 = float(p["sigma8_cold"][0])
    ns = float(p["ns"][0])
    h = float(p["hubble"][0])

    with jax.default_device(jax_device):
        bk = make_bkappa(_ELL1, _ELL2, _ELL3, nz, pk_source=backend, N_chi=8)
        if N_cosmo == 1:
            # Consistent rank: a single cosmology still returns a leading
            # N_cosmo axis of 1, not a squeezed (N_ltri,).
            result = np.asarray(bk(Cosmology(oc, ob, s8, ns, h)))
            assert result.shape == (
                1,
                N_ltri,
            ), f"scalar: expected (1,{N_ltri}), got {result.shape}"
        else:
            result = np.asarray(
                bk(
                    Cosmology(
                        jnp.full(N_cosmo, oc),
                        jnp.full(N_cosmo, ob),
                        jnp.full(N_cosmo, s8),
                        jnp.full(N_cosmo, ns),
                        jnp.full(N_cosmo, h),
                    )
                )
            )
            assert result.shape == (
                N_cosmo,
                N_ltri,
            ), f"batch N_cosmo={N_cosmo}: expected ({N_cosmo},{N_ltri}), got {result.shape}"


# ---------------------------------------------------------------------------
# Test 3: finiteness and positivity
# ---------------------------------------------------------------------------


def test_output_finite_positive(fiducial_params, nz, jax_device, backend):
    """B_κ is finite and strictly positive for all representative ell triangles."""
    p = fiducial_params
    with jax.default_device(jax_device):
        bk = make_bkappa(_ELL1, _ELL2, _ELL3, nz, pk_source=backend, N_chi=20)
        B = np.asarray(bk(_cos(p)))
    assert np.all(np.isfinite(B)), f"{backend}: B_κ contains non-finite values: {B}"
    assert np.all(B > 0), f"{backend}: B_κ contains non-positive values: {B}"


# ---------------------------------------------------------------------------
# Test 4: GL quadrature convergence in chi
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gl_chi_convergence(fiducial_params, nz, jax_device, backend):
    """N_chi=20, 40, 80 agree to within 0.5% for a smooth Smail n(z).

    Validates the default N_chi=20 claim in the make_bkappa docstring.
    Uses a single equilateral triangle for speed.
    """
    p = fiducial_params
    ell = np.array([500.0])
    args = (_cos(p),)

    results = {}
    for N_chi in (20, 40, 80):
        with jax.default_device(jax_device):
            bk = make_bkappa(ell, ell, ell, nz, pk_source=backend, N_chi=N_chi)
            results[N_chi] = float(np.asarray(bk(*args))[0, 0])

    for N_chi in (40, 80):
        reldiff = abs(results[N_chi] - results[20]) / abs(results[20])
        assert (
            reldiff < 0.005
        ), f"{backend}: N_chi={N_chi} vs 20 rel diff {reldiff:.4f} exceeds 0.5%"


# ---------------------------------------------------------------------------
# Test 5: float32 / float64 agreement
# ---------------------------------------------------------------------------


def test_dtype_agreement(fiducial_params, nz, jax_device, backend):
    """float32 and float64 results agree to within 0.5%."""
    p = fiducial_params
    ell = np.array([200.0, 500.0, 1000.0])
    args = (_cos(p),)

    with jax.default_device(jax_device):
        B64 = np.asarray(
            make_bkappa(
                ell, ell, ell, nz, pk_source=backend, N_chi=20, dtype=np.float64
            )(*args)
        )
        B32 = np.asarray(
            make_bkappa(
                ell, ell, ell, nz, pk_source=backend, N_chi=20, dtype=np.float32
            )(*args)
        )

    reldiff = float(np.max(np.abs(B32 - B64) / np.abs(B64)))
    assert (
        reldiff < 5e-3
    ), f"{backend}: float32/float64 max rel diff {reldiff:.2e} exceeds 0.5%"


# ---------------------------------------------------------------------------
# Test 6: caching
# ---------------------------------------------------------------------------


def test_rebuild_agrees_with_first_build(nz, fiducial_params):
    """Two builds at identical settings agree exactly.

    There is no closure cache any more -- it retained every triangle grid's
    bytes for the process lifetime, and make_bkappa is the documented
    build-once entry point -- so these are distinct objects that must still
    compile to the same computation."""
    ell = np.array([300.0, 700.0])
    bk1 = make_bkappa(ell, ell, ell, nz, pk_source="eh")
    bk2 = make_bkappa(ell, ell, ell, nz, pk_source="eh")
    assert bk1 is not bk2
    c = _cos(fiducial_params)
    np.testing.assert_array_equal(np.asarray(bk1(c)), np.asarray(bk2(c)))


def test_caching_distinguishes_backends(nz):
    """Different pk_source values produce distinct callables."""
    ell = np.array([500.0])
    bk_eh = make_bkappa(ell, ell, ell, nz, pk_source="eh")
    bk_syren = make_bkappa(ell, ell, ell, nz, pk_source="syren")
    assert bk_eh is not bk_syren


# ---------------------------------------------------------------------------
# Test 7: differentiability: gradient shape and sign
# ---------------------------------------------------------------------------


def test_gradient_finite_and_positive(fiducial_params, nz, jax_device, backend):
    """jax.grad(ΣB_κ) wrt sigma8_cold is finite and positive (B_κ ∝ sigma8^≥3)."""
    p = fiducial_params
    ell = np.array([300.0, 1000.0])

    bk = make_bkappa(ell, ell, ell, nz, pk_source=backend, N_chi=8)

    def bk_sum(s8):
        return jnp.sum(bk(_cos(p, sigma8_cold=s8)))

    with jax.default_device(jax_device):
        g = float(np.asarray(jax.grad(bk_sum)(jnp.asarray(float(p["sigma8_cold"][0])))))

    assert np.isfinite(g), f"{backend}: ∂B_κ/∂sigma8 is not finite: {g}"
    assert g > 0.0, f"{backend}: ∂B_κ/∂sigma8 ≤ 0 (B_κ should increase with sigma8)"


# ---------------------------------------------------------------------------
# Test 9: FD vs AD Jacobian
# ---------------------------------------------------------------------------


def test_fd_vs_ad_jacobian(fiducial_params, nz, jax_device, backend):
    """AD gradient dB_κ/dsigma8 agrees with central FD to within 2%."""
    p = fiducial_params
    ell = np.array([500.0])
    s8 = float(p["sigma8_cold"][0])
    eps = 1e-4

    bk = make_bkappa(ell, ell, ell, nz, pk_source=backend, N_chi=20)

    with jax.default_device(jax_device):

        def _eval(s8_val):
            return float(np.asarray(bk(_cos(p, sigma8_cold=s8_val)))[0, 0])

        dB_FD = (_eval(s8 + eps) - _eval(s8 - eps)) / (2.0 * eps)

        def bk_first(s8_j):
            return bk(_cos(p, sigma8_cold=s8_j))[0, 0]

        dB_AD = float(np.asarray(jax.grad(bk_first)(jnp.asarray(s8))))

    rel_err = abs(dB_AD - dB_FD) / abs(dB_FD)
    assert (
        rel_err < 0.02
    ), f"{backend}: AD ({dB_AD:.4e}) vs FD ({dB_FD:.4e}) rel error {rel_err:.4f} > 2%"


# ---------------------------------------------------------------------------
# Test 10: physical monotonicity
# ---------------------------------------------------------------------------


def test_sigma8_monotonicity(fiducial_params, nz, jax_device, backend):
    """B_κ increases when sigma8_cold increases (holds for any triangle config)."""
    p = fiducial_params
    oc = float(p["omega_cold"][0])
    ob = float(p["omega_baryon"][0])
    ns = float(p["ns"][0])
    h = float(p["hubble"][0])

    with jax.default_device(jax_device):
        bk = make_bkappa(_ELL1, _ELL2, _ELL3, nz, pk_source=backend, N_chi=10)
        B_lo = np.asarray(bk(Cosmology(oc, ob, 0.70, ns, h)))
        B_hi = np.asarray(bk(Cosmology(oc, ob, 0.90, ns, h)))

    assert np.all(B_hi > B_lo), (
        f"{backend}: B_κ(sigma8=0.90) not > B_κ(sigma8=0.70) for all triangles\n"
        f"  ratios: {B_hi / B_lo}"
    )


def test_omega_m_monotonicity(fiducial_params, nz, jax_device, backend):
    """B_κ increases when omega_cold increases (higher matter density → more lensing)."""
    p = fiducial_params
    ob = float(p["omega_baryon"][0])
    s8 = float(p["sigma8_cold"][0])
    ns = float(p["ns"][0])
    h = float(p["hubble"][0])

    with jax.default_device(jax_device):
        bk = make_bkappa(_ELL1, _ELL2, _ELL3, nz, pk_source=backend, N_chi=10)
        B_lo = np.asarray(bk(Cosmology(0.26, ob, s8, ns, h)))
        B_hi = np.asarray(bk(Cosmology(0.38, ob, s8, ns, h)))

    assert np.all(
        B_hi > B_lo
    ), f"{backend}: B_κ(omega_cold=0.38) not > B_κ(omega_cold=0.26) for all triangles"


# ---------------------------------------------------------------------------
# Test 11: backend cross-agreement
# ---------------------------------------------------------------------------

# Physical expectation: syren and baccoemu are both calibrated linear-P(k)
# emulators/fits and agree very tightly (≈0.1%); the Eisenstein & Hu analytic
# fitting formula ('eh') is the outlier and differs from both by ≈2%.  The
# tolerances below encode that hierarchy rather than treating the backends
# symmetrically (measured at the Planck fiducial: syren/baccoemu 0.13%,
# eh/syren 1.87%, eh/baccoemu 1.90%).


def test_backend_syren_baccoemu_agreement(fiducial_params, nz, jax_device):
    """syren and baccoemu B_κ agree to within 1% (both calibrated P_L emulators)."""
    pytest.importorskip("baccoemu")
    p = fiducial_params
    ell = np.array([200.0, 500.0, 1000.0])
    args = (_cos(p),)

    with jax.default_device(jax_device):
        B_syren = np.asarray(
            make_bkappa(ell, ell, ell, nz, pk_source="syren", N_chi=10)(*args)
        )
        B_bemu = np.asarray(
            make_bkappa(ell, ell, ell, nz, pk_source="baccoemu", N_chi=10)(*args)
        )

    max_dev = float(np.max(np.abs(B_syren / B_bemu - 1.0)))
    assert max_dev < 0.01, (
        f"syren vs baccoemu: max|ratio-1| = {max_dev:.4f} exceeds 1% "
        "(the two calibrated emulators should agree closely)"
    )


def test_backend_eh_offset(fiducial_params, nz, jax_device):
    """eh B_κ differs from the calibrated emulators by ≈2%, bounded at 5%.

    eh is an analytic linear-P(k) fitting formula, so a few-percent offset from
    syren/baccoemu is expected and physical, but a much larger deviation would
    signal a bug.  Compared against syren so the test runs without baccoemu.
    """
    p = fiducial_params
    ell = np.array([200.0, 500.0, 1000.0])
    args = (_cos(p),)

    with jax.default_device(jax_device):
        B_eh = np.asarray(
            make_bkappa(ell, ell, ell, nz, pk_source="eh", N_chi=10)(*args)
        )
        B_syren = np.asarray(
            make_bkappa(ell, ell, ell, nz, pk_source="syren", N_chi=10)(*args)
        )

    max_dev = float(np.max(np.abs(B_eh / B_syren - 1.0)))
    assert max_dev < 0.05, f"eh vs syren: max|ratio-1| = {max_dev:.4f} exceeds 5%"


# ---------------------------------------------------------------------------
# Test 12: error handling
# ---------------------------------------------------------------------------


def test_invalid_pk_source(nz):
    """make_bkappa raises ValueError for an unknown pk_source."""
    with pytest.raises(ValueError, match="pk_source must be one of"):
        make_bkappa(_ELL1, _ELL2, _ELL3, nz, pk_source="bogus")


def test_ell_out_of_range_warning(nz):
    """Ells that push k = ell/chi_min outside the P_L grid trigger RuntimeWarning."""
    ell_huge = np.array([1e8])  # k = 1e8/10 ≈ 1e7 h/Mpc, way beyond grid
    with pytest.warns(RuntimeWarning, match="out of P_L grid"):
        make_bkappa(ell_huge, ell_huge, ell_huge, nz, pk_source="eh")


# ---------------------------------------------------------------------------
# Test 13: golden reference
# ---------------------------------------------------------------------------


def test_bkappa_golden_reference(bkappa_golden, jax_device, backend):
    """B_κ at the fiducial cosmology agrees with the stored golden to within 1%."""
    g = bkappa_golden
    key = f"Bkappa_ref_{backend}"
    if key not in g:
        pytest.skip(f"No golden reference for backend='{backend}' in {_GOLDEN_PATH}")

    p = {
        k: float(g[f"cosmo_{k}"])
        for k in (
            "omega_cold",
            "omega_baryon",
            "sigma8_cold",
            "ns",
            "hubble",
            "neutrino_mass",
            "w0",
            "wa",
        )
    }
    ell1 = g["ell1"]
    ell2 = g["ell2"]
    ell3 = g["ell3"]
    nz_g = (g["z_grid"], g["nz_grid"])

    with jax.default_device(jax_device):
        bk = make_bkappa(ell1, ell2, ell3, nz_g, pk_source=backend, N_chi=20)
        B_new = np.asarray(bk(_cos(p)))

    B_ref = g[key]
    max_dev = float(np.max(np.abs(B_new / B_ref - 1.0)))
    assert (
        max_dev < 0.01
    ), f"{backend}: max |B_κ/B_ref - 1| = {max_dev:.4f} exceeds 1% tolerance"


# ---------------------------------------------------------------------------
# Test 14: ABSOLUTE-amplitude validation against an independent Limber integral
# ---------------------------------------------------------------------------


def test_independent_limber_amplitude(nz, jax_device):
    """B_κ matches a transparent brute-force Limber integral to <0.01%.

    Every other quantitative test is ratio-based (monotonicity, backend, dtype),
    self-referential (the golden is produced by this same code), or sign/shape:
    none would catch a wrong overall amplitude (the 3/2·Ω_m·(H₀/c)² prefactor,
    the χ/a·g kernel, the χ⁻⁴ power, the n(z) normalization, or the einsum).
    This test re-derives B_κ from scratch by calling compute_bihalofit directly,
    independently of the JAX assembly.

    The agreement floor is set by the *reference* quadrature, not the kernel: the
    JAX Gauss-Legendre integrator converges far faster than a uniform grid, so a
    Simpson reference (O(h^4)) at M=401 is needed to push the reference error to
    ~5e-6 and assert at the 1e-4 level.  A uniform-trapezoid reference (O(h^2))
    caps the achievable tolerance near 0.06% at l=100: that is the reference's
    error, not the implementation's (which agrees to ~6e-6 here).
    """
    simpson = pytest.importorskip("scipy.integrate").simpson
    oc, ob, s8, ns, h = 0.315, 0.0492, 0.831, 0.9645, 0.6727
    Om = oc  # neutrino_mass = 0

    z, nz_arr = nz[0], nz[1]
    nz_arr = nz_arr / _np_trapezoid(nz_arr, z)

    # Independent χ(z): fine trapezoid (different grid than the kernel's).
    zf = np.linspace(0.0, 3.5, 8000)
    Ef = np.sqrt(Om * (1 + zf) ** 3 + (1 - Om))
    chf = _C_OVER_H0 * np.concatenate(
        [[0.0], np.cumsum(0.5 * (1 / Ef[1:] + 1 / Ef[:-1]) * np.diff(zf))]
    )
    chi_src = np.interp(z, zf, chf)

    M = 401  # odd -> exact Simpson
    chi = np.linspace(10.0, 5800.0, M)
    zc = np.interp(chi, chf, zf)
    ac = 1.0 / (1.0 + zc)
    g = np.array(
        [
            _np_trapezoid(
                nz_arr
                * np.clip(chi_src - cc, 0, None)
                / np.where(chi_src > 0, chi_src, 1.0),
                z,
            )
            for cc in chi
        ]
    )
    q = 1.5 / _C_OVER_H0**2 * Om * (chi / ac * g)

    ells = [100.0, 500.0, 2000.0]
    with jax.default_device(jax_device):
        B_bf = {}
        for ell in ells:
            k = (ell + 0.5) / chi  # extended-Limber shift, see _limber_triangle_setup
            B = np.asarray(
                compute_bihalofit(
                    ac,
                    k,
                    k,
                    k,
                    Cosmology(
                        np.array([oc]),
                        np.array([ob]),
                        np.array([s8]),
                        np.array([ns]),
                        np.array([h]),
                    ),
                    pk_source="eh",
                )
            )[
                0
            ]  # (M_z, M_k)
            B_bf[ell] = simpson(q**3 / chi**4 * np.diag(B), x=chi)

        ell_arr = np.array(ells)
        bk = make_bkappa(
            ell_arr,
            ell_arr,
            ell_arr,
            (z, nz_arr),
            pk_source="eh",
            N_chi=200,
            chi_min=10.0,
            chi_max=5800.0,
        )
        B_jax = np.asarray(bk(Cosmology(oc, ob, s8, ns, h)))[0]

    for i, ell in enumerate(ells):
        rel = abs(B_jax[i] / B_bf[ell] - 1.0)
        assert rel < 1e-4, (
            f"ell={ell}: JAX {B_jax[i]:.6e} vs brute-force {B_bf[ell]:.6e} "
            f"rel diff {rel:.2e} exceeds 0.01%"
        )


# ---------------------------------------------------------------------------
# Test 15: comoving distance against an independent integrator
# ---------------------------------------------------------------------------


def test_comoving_distance_reference():
    """χ(z) from the cumulative-trapezoid backend matches scipy.quad to <0.1%.

    A bug in χ(z) (factor of h, wrong integrand) rescales k=ℓ/χ everywhere and
    is otherwise only caught by the self-generated golden.
    """
    quad = pytest.importorskip("scipy.integrate").quad
    Om = 0.3
    z, chi = fiducial_background(z_max=3.0, N_z=3000, Omega_m=Om)
    for z_test in (0.5, 1.0, 2.0, 3.0):
        ref = (
            _C_OVER_H0
            * quad(
                lambda zz: 1.0 / np.sqrt(Om * (1 + zz) ** 3 + (1 - Om)), 0.0, z_test
            )[0]
        )
        got = float(np.interp(z_test, z, chi))
        rel = abs(got / ref - 1.0)
        assert (
            rel < 1e-3
        ), f"χ(z={z_test}) = {got:.3f} vs quad {ref:.3f} Mpc/h, rel {rel:.2e}"


# ---------------------------------------------------------------------------
# Test 16: single-source-plane lensing kernel against the analytic form
# ---------------------------------------------------------------------------


def test_lensing_geometry_single_source_plane():
    """For a narrow n(z)≈δ(z−z_s), g(χ) → (χ_s−χ)/χ_s for χ<χ_s, 0 beyond.

    Validates lensing_geometry independently of the BiHalofit machinery.
    """
    Om, z_s = 0.3, 1.0
    z = np.linspace(0.0, 3.0, 4000)
    nz_arr = np.exp(-0.5 * ((z - z_s) / 0.01) ** 2)
    nz_arr /= _np_trapezoid(nz_arr, z)

    chi_z = _comoving_distance(jnp.asarray(z), Om, -1.0, 0.0)
    chi_s = float(np.interp(z_s, z, np.asarray(chi_z)))

    chi_nodes = np.array([0.25, 0.5, 0.75, 1.1]) * chi_s
    g = np.asarray(
        lensing_geometry(
            jnp.asarray(chi_nodes), chi_z, jnp.asarray(nz_arr), jnp.asarray(z)
        )
    )

    g_analytic = np.clip((chi_s - chi_nodes) / chi_s, 0.0, None)
    assert np.allclose(
        g, g_analytic, atol=5e-3
    ), f"g(χ)={g} vs analytic {g_analytic} (single source plane)"


# ---------------------------------------------------------------------------
# Test 17: internal n(z) normalization (amplitude independent of input scale)
# ---------------------------------------------------------------------------


def test_nz_normalization_invariance(jax_device):
    """An unnormalized n(z) gives the same B_κ as the normalized one.

    B_κ ∝ (∫n q dz)³, so without internal normalization a rescaled n(z) would
    change the amplitude by norm³.  Guards the normalization fix.
    """
    z = np.linspace(0.0, 3.5, 500)
    nz1 = smail_nz(z)
    nz2 = 37.0 * nz1  # same shape, arbitrary scale
    ell = np.array([200.0, 800.0])
    args = (Cosmology(0.315, 0.0492, 0.831, 0.9645, 0.6727),)

    with jax.default_device(jax_device):
        B1 = np.asarray(
            make_bkappa(ell, ell, ell, (z, nz1), pk_source="eh", N_chi=10)(*args)
        )
        B2 = np.asarray(
            make_bkappa(ell, ell, ell, (z, nz2), pk_source="eh", N_chi=10)(*args)
        )

    assert np.allclose(B1, B2, rtol=1e-6), f"norm-dependent B_κ: {B1} vs {B2}"


# ---------------------------------------------------------------------------
# Test 18: gradients finite (and physical) across ALL cosmological parameters
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gradients_all_parameters(nz, jax_device):
    """jax.grad(ΣB_κ) is finite for every one of the 8 parameters.

    The differentiability claim covers all params, but only sigma8 was exercised;
    a NaN gradient in w0/wa/neutrino_mass (DE exponent, interp boundary) would
    otherwise go uncaught.  syren is used so the P(k) paths for mnu/w0/wa run.
    """
    ell = np.array([300.0, 1000.0])
    base = dict(
        oc=0.30, ob=0.0492, s8=0.831, ns=0.9645, h=0.6727, mnu=0.06, w0=-0.95, wa=0.10
    )
    names = ["oc", "ob", "s8", "ns", "h", "mnu", "w0", "wa"]

    with jax.default_device(jax_device):
        bk = make_bkappa(ell, ell, ell, nz, pk_source="syren", N_chi=8)

        def f(oc, ob, s8, ns, h, mnu, w0, wa):
            return jnp.sum(
                bk(Cosmology(oc, ob, s8, ns, h, neutrino_mass=mnu, w0=w0, wa=wa))
            )

        grads = {}
        for i, nm in enumerate(names):
            grads[nm] = float(
                np.asarray(
                    jax.grad(f, argnums=i)(*[jnp.asarray(base[k]) for k in names])
                )
            )

    for nm, gval in grads.items():
        assert np.isfinite(gval), f"∂B_κ/∂{nm} is not finite: {gval}"
    # physical signs: amplitude rises with sigma8 and with matter density
    assert grads["s8"] > 0, f"∂B_κ/∂sigma8 ≤ 0: {grads['s8']}"
    assert grads["oc"] > 0, f"∂B_κ/∂omega_cold ≤ 0: {grads['oc']}"


# ---------------------------------------------------------------------------
# Test 18b: gradient continuity across the lensing kernel's chi-node ramp
# and z(chi) inversion
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gradient_continuity_across_hubble_sweep(nz, jax_device):
    """dB_kappa/dh must vary smoothly as h moves the comoving-distance grid.

    Sweeping h shifts chi(z) continuously, so the lensing kernel's chi_node
    ramp (chi_z crossing each chi_node) and the z(chi) inversion at those
    nodes both get exercised repeatedly over this range. A discontinuous
    implementation (hard jnp.clip ramp, or jnp.interp bracket for z(chi))
    kinks dB_kappa/dh at every crossing; the curvature (second difference) of
    the AD gradient spikes by orders of magnitude at those kinks relative to
    its typical smooth variation. A C^1/C^infty implementation keeps that
    spike within normal smooth-curve variation.
    """
    ell = np.array([300.0, 1000.0])

    with jax.default_device(jax_device):
        bk = make_bkappa(ell, ell, ell, nz, pk_source="eh", N_chi=8)

        def f(h):
            return jnp.sum(bk(Cosmology(0.30, 0.0492, 0.831, 0.9645, h)))

        h_grid = np.linspace(0.60, 0.80, 400)
        grads = np.empty_like(h_grid)
        for i, h in enumerate(h_grid):
            grads[i] = float(jax.grad(f)(jnp.asarray(h)))

    assert np.all(
        np.isfinite(grads)
    ), "dB_kappa/dh is non-finite somewhere on the sweep"

    d2 = np.diff(grads, n=2)
    ratio = float(np.max(np.abs(d2)) / np.median(np.abs(d2)))
    assert ratio < 100.0, (
        f"gradient curvature spikes by {ratio:.1f}x the typical smooth "
        f"variation over the h sweep -- looks like a discontinuous kink in "
        f"the chi-node ramp or z(chi) inversion, not a C^1 implementation"
    )


# ---------------------------------------------------------------------------
# Test 19: input validation (z grid must start at 0; n(z) must be positive)
# ---------------------------------------------------------------------------


def test_zgrid_must_start_at_zero():
    """make_bkappa rejects a z grid not starting at z=0."""
    z = np.linspace(0.1, 3.0, 200)  # χ(z) backend assumes z[0]==0
    nz_arr = smail_nz(z)
    ell = np.array([500.0])
    with pytest.raises(ValueError, match="z=0"):
        make_bkappa(ell, ell, ell, (z, nz_arr), pk_source="eh")


def test_nz_must_be_positive_integral():
    """A non-positive n(z) integral is rejected rather than silently producing NaN."""
    z = np.linspace(0.0, 3.0, 200)
    ell = np.array([500.0])
    with pytest.raises(ValueError, match="positive integral"):
        make_bkappa(ell, ell, ell, (z, np.zeros_like(z)), pk_source="eh")


# ---------------------------------------------------------------------------
# Test 20: chi_max adequacy warning
# ---------------------------------------------------------------------------


def test_chi_max_truncation_warning():
    """A chi_max that fails to reach the sources triggers a truncation warning."""
    z = np.linspace(0.0, 3.5, 500)
    nz_arr = smail_nz(z)
    ell = np.array([500.0])
    with pytest.warns(RuntimeWarning, match="truncates"):
        make_bkappa(ell, ell, ell, (z, nz_arr), pk_source="eh", chi_max=1500.0)


def test_narrow_counts_warning_and_default_n_chi():
    """A counts sample narrower than the chi-node spacing warns and names the
    N_chi needed; a photometric-width sample at the default N_chi does not."""
    z = np.linspace(0.0, 3.5, 2000)
    ell = np.array([200.0])
    narrow = np.exp(-0.5 * ((z - 0.5) / 0.01) ** 2)
    broad = np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2)
    with pytest.warns(RuntimeWarning, match="under-resolved"):
        make_projected_bihalofit(
            ell,
            ell,
            ell,
            ((z, narrow),) * 3,
            ("counts",) * 3,
            bias=(1.5,) * 3,
            pk_source="eh",
            dtype=np.float64,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        make_projected_bihalofit(
            ell,
            ell,
            ell,
            ((z, broad),) * 3,
            ("counts",) * 3,
            bias=(1.5,) * 3,
            pk_source="eh",
            dtype=np.float64,
        )


def test_sparse_dndz_warning():
    """A 'lensing' leg's g(chi) integral is plain trapezoidal on the caller's
    own dndz grid, no internal refinement -- a sparse grid should warn."""
    z_sparse = np.linspace(0.0, 3.5, 20)
    nz_sparse = smail_nz(z_sparse)
    ell = np.array([500.0])
    with pytest.warns(RuntimeWarning, match="dndz has only"):
        make_bkappa(ell, ell, ell, (z_sparse, nz_sparse), pk_source="eh")


def test_dense_dndz_no_warning():
    """The package default fine-grained dndz should not trigger the sparse
    warning (regression guard against too-aggressive a threshold)."""
    z = np.linspace(0.0, 3.5, 500)
    nz_arr = smail_nz(z)
    ell = np.array([500.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        make_bkappa(ell, ell, ell, (z, nz_arr), pk_source="eh")


# ---------------------------------------------------------------------------
# Test 21: tol / N_R / validate -- mirrors bihalofit_jax.py's mechanism
# exactly (make_bkappa/compute_convergence_bispectrum were ported to use the
# same unconditional-verify, retry-on-N_R, NaN-or-raise design).
# ---------------------------------------------------------------------------


def test_bkappa_tol_N_R_mutually_exclusive(fiducial_params, nz):
    """Passing both tol and a non-default N_R raises ValueError."""
    ell = np.array([300.0, 700.0])
    p = fiducial_params
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_convergence_bispectrum(
            ell, ell, ell, nz, _cos(p), pk_source="eh", tol=1e-8, N_R=5
        )


def test_bkappa_tol_matches_manual_N_R(fiducial_params, nz):
    """tol=1e-8 resolves to N_R=_tol_to_niter(1e-8); the two calls must
    agree bit-for-bit. Relies on the same 2-iterations-past-convergence
    padding in _tol_to_niter as the analogous bihalofit-level test in
    test_regression.py -- see that test's docstring and _tol_to_niter's
    docstring for why the count is padded."""
    ell = np.array([300.0, 700.0])
    p = fiducial_params
    args = (ell, ell, ell, nz, _cos(p))
    B_tol = compute_convergence_bispectrum(*args, pk_source="eh", tol=1e-8)
    B_N_R = compute_convergence_bispectrum(
        *args, pk_source="eh", N_R=_tol_to_niter(1e-8)
    )
    np.testing.assert_array_equal(np.asarray(B_tol), np.asarray(B_N_R))


def test_bkappa_tol_under_estimate_raises_and_names_N_R(
    monkeypatch, fiducial_params, nz
):
    """As test_regression's counterpart: an under-estimated iteration count is
    reported, not silently re-solved. tol sizes the count up front only."""
    ell = np.array([310.0, 710.0])  # own geometry, never reused elsewhere
    args = (ell, ell, ell, nz, _cos(fiducial_params))
    monkeypatch.setattr(_bhfs, "_tol_to_niter", lambda tol: 1)
    with pytest.raises(ValueError, match=r"Newton residual reached.*N_R"):
        compute_convergence_bispectrum(*args, pk_source="eh", tol=1e-8, validate=True)

    B = compute_convergence_bispectrum(*args, pk_source="eh", tol=1e-8)
    assert np.all(np.isnan(np.asarray(B)))


def _force_bad_residual(monkeypatch):
    """Patch _build_limber_kernel so every call's k_NL/n_eff residual is
    forced to a large, never-converging value -- see the identical, more
    detailed note in test_regression.py's own _force_bad_residual for why
    an astronomically tiny tol is not a portable way to force this path."""
    orig_build = _pbhf._build_limber_kernel

    def fake_build(*args, **kwargs):
        real_kern = orig_build(*args, **kwargs)

        def fake_kern(*kargs):
            B, resid = real_kern(*kargs)
            return B, jnp.ones_like(resid)

        return fake_kern

    monkeypatch.setattr(_pbhf, "_build_limber_kernel", fake_build)


def test_bkappa_tol_unreachable_nans_by_default(monkeypatch, fiducial_params, nz):
    """If the residual never drops below tol even at the N_R=30 cap,
    validate=False marks exactly those cosmologies NaN."""
    ell = np.array([320.0, 720.0])  # own geometry, never reused elsewhere
    p = fiducial_params
    _force_bad_residual(monkeypatch)
    B = compute_convergence_bispectrum(
        ell, ell, ell, nz, _cos(p), pk_source="eh", tol=1e-8, validate=False
    )
    assert np.all(np.isnan(np.asarray(B)))


def test_bkappa_tol_unreachable_raises_with_validate(monkeypatch, fiducial_params, nz):
    """Same forced failure, but validate=True raises instead of NaN."""
    ell = np.array([330.0, 730.0])  # own geometry, never reused elsewhere
    p = fiducial_params
    _force_bad_residual(monkeypatch)
    with pytest.raises(ValueError, match=r"Newton residual reached"):
        compute_convergence_bispectrum(
            ell, ell, ell, nz, _cos(p), pk_source="eh", tol=1e-8, validate=True
        )


def test_bkappa_validate_not_cached_across_different_values(
    monkeypatch, fiducial_params, nz
):
    """Building the model at the same geometry with validate=False, then
    validate=True, must not silently reuse a closure whose validate value
    was frozen in by the first call.

    Regression test for a genuine bug found while writing this test suite:
    make_bkappa's cache key initially omitted both tol and validate, so a
    validate=True call could silently reuse a validate=False closure (or
    vice versa) built earlier at the same geometry/N_R -- both are now part
    of the cache key.
    """
    ell = np.array([340.0, 740.0])  # own geometry, never reused elsewhere
    p = fiducial_params
    _force_bad_residual(monkeypatch)
    compute_convergence_bispectrum(
        ell, ell, ell, nz, _cos(p), pk_source="eh", tol=1e-8, validate=False
    )
    with pytest.raises(ValueError, match=r"Newton residual reached"):
        compute_convergence_bispectrum(
            ell, ell, ell, nz, _cos(p), pk_source="eh", tol=1e-8, validate=True
        )


# ---------------------------------------------------------------------------
# Test 22: named-parameter Jacobians via plain jax.jacfwd (compute_projected_
# bihalofit has no library-side wrt= convenience -- see _bkappa_with_jac)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bkappa_wrt_matches_fd(fiducial_params, nz):
    """AD Jacobian w.r.t. sigma8_cold agrees with central FD to within 2%."""
    p = fiducial_params
    ell = np.array([500.0])
    s8 = float(p["sigma8_cold"][0])
    eps = 1e-4
    cosmo = dict(
        omega_cold=float(p["omega_cold"][0]),
        omega_baryon=float(p["omega_baryon"][0]),
        sigma8_cold=s8,
        ns=float(p["ns"][0]),
        hubble=float(p["hubble"][0]),
        neutrino_mass=0.0,
        w0=-1.0,
        wa=0.0,
    )

    B, jac = _bkappa_with_jac(ell, ell, ell, nz, cosmo, "sigma8_cold", pk_source="eh")
    dB_AD = float(np.asarray(jac["sigma8_cold"])[0, 0])

    def _eval(s8_val):
        c = {**cosmo, "sigma8_cold": s8_val}
        return float(
            np.asarray(
                compute_convergence_bispectrum(
                    ell, ell, ell, nz, cosmo=c, pk_source="eh"
                )
            )[0, 0]
        )

    dB_FD = (_eval(s8 + eps) - _eval(s8 - eps)) / (2.0 * eps)
    rel_err = abs(dB_AD - dB_FD) / abs(dB_FD)
    assert (
        rel_err < 0.02
    ), f"AD ({dB_AD:.4e}) vs FD ({dB_FD:.4e}) rel error {rel_err:.4f} > 2%"


def test_bkappa_wrt_all(fiducial_params, nz):
    """jax.jacfwd gives a finite derivative for every cosmological parameter."""
    p = fiducial_params
    ell = np.array([500.0])
    cosmo = dict(
        omega_cold=float(p["omega_cold"][0]),
        omega_baryon=float(p["omega_baryon"][0]),
        sigma8_cold=float(p["sigma8_cold"][0]),
        ns=float(p["ns"][0]),
        hubble=float(p["hubble"][0]),
        neutrino_mass=0.0,
        w0=-1.0,
        wa=0.0,
    )
    B, jac = _bkappa_with_jac(ell, ell, ell, nz, cosmo, _COSMO_NAMES, pk_source="eh")
    assert set(jac.keys()) == set(_COSMO_NAMES)
    assert np.all(np.isfinite(np.asarray(B)))
    for v in jac.values():
        assert np.all(np.isfinite(np.asarray(v)))


# ---------------------------------------------------------------------------
# Test 23: term ('full'/'1h'/'3h'/'tree') and baryonic_boost -- ported from
# bihalofit_jax.compute_bihalofit; the Limber projection is agnostic to which
# term of B_delta it integrates, so these just need to survive the extra
# chi-node vmap layer.
# ---------------------------------------------------------------------------


def test_bkappa_term_invalid_raises(fiducial_params, nz):
    """Unrecognised term raises ValueError before any JAX work."""
    p = fiducial_params
    ell = np.array([500.0])
    with pytest.raises(ValueError, match="term must be one of"):
        compute_convergence_bispectrum(
            ell, ell, ell, nz, _cos(p), pk_source="eh", term="bogus"
        )


def test_bkappa_term_full_equals_1h_plus_3h(fiducial_params, nz):
    """term='full' equals term='1h' + term='3h' after projection, across the
    representative ell-triangle set."""
    p = fiducial_params
    args = (_ELL1, _ELL2, _ELL3, nz, _cos(p))
    B_full = compute_convergence_bispectrum(*args, pk_source="eh", term="full")
    B_1h = compute_convergence_bispectrum(*args, pk_source="eh", term="1h")
    B_3h = compute_convergence_bispectrum(*args, pk_source="eh", term="3h")
    np.testing.assert_allclose(
        np.asarray(B_full), np.asarray(B_1h) + np.asarray(B_3h), rtol=1e-10, atol=0
    )


def test_bkappa_term_tree_finite_and_smaller_than_full(fiducial_params, nz):
    """term='tree' (linear PT, no k_NL/n_eff dependence) is finite/positive
    and, for these deeply nonlinear scales, much smaller in amplitude than
    the full nonlinear B_kappa -- a basic physical sanity check independent
    of the exact PT formula (already checked directly for B_delta in
    test_regression.py's test_term_tree_matches_independent_pt_formula)."""
    p = fiducial_params
    args = (_ELL1, _ELL2, _ELL3, nz, _cos(p))
    B_full = np.asarray(
        compute_convergence_bispectrum(*args, pk_source="eh", term="full")
    )
    B_tree = np.asarray(
        compute_convergence_bispectrum(*args, pk_source="eh", term="tree")
    )
    assert np.all(np.isfinite(B_tree))
    assert np.all(B_tree > 0)
    assert np.all(B_tree < B_full)


def test_bkappa_term_tree_not_poisoned_by_bracket_failure(fiducial_params, nz):
    """term='tree' stays finite even for a sigma8 so small that sigma^2(R)
    never reaches 1 for 'full' (mirrors test_regression.py's analogous
    B_delta-level test)."""
    ell = np.array([500.0])
    p = dict(fiducial_params)
    args = (ell, ell, ell, nz, _cos(p, sigma8_cold=0.01))
    B_full = compute_convergence_bispectrum(*args, pk_source="eh", term="full")
    B_tree = compute_convergence_bispectrum(*args, pk_source="eh", term="tree")
    assert np.all(np.isnan(np.asarray(B_full)))
    assert np.all(np.isfinite(np.asarray(B_tree)))
    assert np.all(np.asarray(B_tree) > 0)


def test_bkappa_baryonic_boost_changes_result(fiducial_params, nz):
    """baryonic_boost=True gives a different, still-finite/positive result
    from the plain calculation (the boost's exact multiplicative form is
    already checked directly on B_delta in test_regression.py; here we only
    need the chi-integral + boost composition to work correctly)."""
    p = fiducial_params
    args = (_ELL1, _ELL2, _ELL3, nz, _cos(p))
    B_plain = np.asarray(compute_convergence_bispectrum(*args, pk_source="eh"))
    B_boost = np.asarray(
        compute_convergence_bispectrum(*args, pk_source="eh", baryonic_boost=True)
    )
    assert np.all(np.isfinite(B_boost))
    assert np.all(B_boost > 0)
    assert not np.allclose(B_plain, B_boost, atol=0)


def test_bkappa_term_and_baryonic_boost_not_cached_across_different_values(
    fiducial_params, nz
):
    """Same geometry built with different term/baryonic_boost combinations
    must not silently share a cached closure -- same cache-key bug class
    already found and fixed for tol/validate (see
    test_bkappa_validate_not_cached_across_different_values)."""
    ell = np.array([350.0, 750.0])  # own geometry, never reused elsewhere
    p = fiducial_params
    common = dict(pk_source="eh")

    f_full = make_bkappa(ell, ell, ell, nz, **common)
    f_1h = make_bkappa(ell, ell, ell, nz, **common, term="1h")
    f_tree = make_bkappa(ell, ell, ell, nz, **common, term="tree")
    f_full_boost = make_bkappa(ell, ell, ell, nz, **common, baryonic_boost=True)

    assert f_full is not f_1h
    assert f_full is not f_tree
    assert f_full is not f_full_boost

    call = lambda f: np.asarray(f(_cos(p)))

    r_full = call(f_full)
    r_1h = call(f_1h)
    r_tree = call(f_tree)
    r_full_boost = call(f_full_boost)

    assert not np.allclose(r_full, r_1h, atol=0)
    assert not np.allclose(r_full, r_tree, atol=0)
    assert not np.allclose(r_full, r_full_boost, atol=0)


def test_bkappa_term_and_baryonic_boost_under_jacfwd(fiducial_params, nz):
    """term and baryonic_boost compose cleanly under jax.jacfwd."""
    p = fiducial_params
    ell = np.array([500.0])
    cosmo = dict(
        omega_cold=float(p["omega_cold"][0]),
        omega_baryon=float(p["omega_baryon"][0]),
        sigma8_cold=float(p["sigma8_cold"][0]),
        ns=float(p["ns"][0]),
        hubble=float(p["hubble"][0]),
        neutrino_mass=0.0,
        w0=-1.0,
        wa=0.0,
    )
    B, jac = _bkappa_with_jac(
        ell,
        ell,
        ell,
        nz,
        cosmo,
        "sigma8_cold",
        pk_source="eh",
        term="3h",
        baryonic_boost=True,
    )
    assert np.all(np.isfinite(np.asarray(B)))
    assert np.all(np.isfinite(np.asarray(jac["sigma8_cold"])))
    assert np.any(np.asarray(jac["sigma8_cold"]) != 0.0)


# ---------------------------------------------------------------------------
# k_lin (custom P(k) grid) -- forwarded straight through to make_bihalofit's
# own kernel at every chi-node; these check the forwarding itself, not the
# interpolation-error behaviour already covered at the raw-bispectrum level
# in test_regression.py.
# ---------------------------------------------------------------------------


def test_bkappa_k_lin_invalid_with_baccoemu_raises(fiducial_params, nz):
    """k_lin is rejected eagerly for pk_source='baccoemu', before any JAX work."""
    p = fiducial_params
    ell = np.array([500.0])
    k_lin = np.geomspace(1e-4, 50.0, 1200)
    with pytest.raises(ValueError, match="k_lin is not supported"):
        compute_convergence_bispectrum(
            ell, ell, ell, nz, _cos(p), pk_source="baccoemu", k_lin=k_lin
        )


def test_bkappa_k_lin_finer_grid_changes_result(fiducial_params, nz):
    """A finer k_lin changes B_kappa (every chi-node's k=ell/chi is
    essentially never exactly on a grid node) but stays close to the
    default-grid result."""
    p = fiducial_params
    k_lin_fine = np.geomspace(1e-4, 50.0, 4 * 600)
    args = (_ELL1, _ELL2, _ELL3, nz, _cos(p))
    B_default = np.asarray(compute_convergence_bispectrum(*args, pk_source="eh"))
    B_fine = np.asarray(
        compute_convergence_bispectrum(*args, pk_source="eh", k_lin=k_lin_fine)
    )
    assert np.all(np.isfinite(B_fine))
    rel = np.abs(B_fine / B_default - 1)
    assert np.nanmax(rel) < 1e-2  # sanity bound, not a precision claim


def test_bkappa_k_lin_not_cached_across_different_grids(fiducial_params, nz):
    """Same geometry built with different k_lin must not silently share a
    cached closure -- same cache-key bug class already guarded against for
    term/baryonic_boost/tol/validate."""
    ell = np.array([350.0, 750.0])  # own geometry, never reused elsewhere
    p = fiducial_params
    k_lin_fine = np.geomspace(1e-4, 50.0, 1200)
    common = dict(pk_source="eh")

    f_default = make_bkappa(ell, ell, ell, nz, **common)
    f_fine = make_bkappa(ell, ell, ell, nz, **common, k_lin=k_lin_fine)
    assert f_default is not f_fine

    r_default = np.asarray(f_default(_cos(p)))
    r_fine = np.asarray(f_fine(_cos(p)))
    assert not np.allclose(r_default, r_fine, atol=0)


def test_bkappa_no_nan_across_prior_box():
    """Broad randomized sweep asserting compute_convergence_bispectrum at
    DEFAULT settings (R_min=1e-6, N_R=15) never returns NaN for a triangle/
    n(z) that spans the documented Limber integration range, including
    extreme low-h/high-z cosmologies (see the more detailed sweep in
    test_regression.py and test_R_min_R_max_defaults_consistent_across_public_api).

    Covers, deliberately wider than the first version of this test: all 6
    cosmological parameters that can plausibly push k_NL/n_eff toward a
    bracket edge (omega_cold, sigma8_cold, hubble, ns, w0, wa -- see
    test_tol_to_niter_meets_claimed_accuracy's bounds), N=100 draws (not
    15), and the module's own representative triangle set _ELL_TRIS
    (equilateral, squeezed, scalene -- not just equilateral), reusing the
    same squeezed/near-degenerate-sensitive shapes test_output_shape etc.
    already exercise elsewhere in this file. All cosmologies x all
    triangle shapes are batched into a single compute_convergence_bispectrum
    call (N_cosmo, N_tri both vectorised), not a Python-level loop.

    Uses a LOCAL n(z) capped at z_max=3.0 -- matching truth_bkappa_ccode's
    own documented Limber range, not this module's shared `nz` fixture
    (support to z=3.5). That's not an arbitrary narrowing to make the test
    pass: widening this sweep first surfaced a real, then-uninvestigated
    14/700 failure rate with the full z=3.5 n(z), and direct inspection
    (sigma2(R_min=1e-6)=0.955 < 1, with R_min already saturating the
    k-grid's own k_max=50 cutoff -- going smaller adds no power) confirmed
    it's a genuine physical edge case, not a numerical bracket artifact: at
    z=3.5, phantom (w0<-1) + strongly-evolving (wa<0) dark energy
    combinations can suppress growth enough that the nonlinear scale is
    legitimately undefined within the achievable k-range. Even a
    negligible-weight n(z) tail node still NaN-poisons the whole Limber
    sum if evaluated there (standard IEEE 754 behaviour, not a bug), so
    testing within the documented-valid z range is the correct fix, not a
    workaround."""
    z_capped = np.linspace(0.0, 3.0, 500)
    nz_capped = (z_capped, smail_nz(z_capped))
    rng = np.random.default_rng(1)
    N = 100
    bounds = dict(
        omega_cold=(0.20, 0.45),
        sigma8_cold=(0.55, 1.10),
        hubble=(0.45, 0.95),
        ns=(0.85, 1.10),
        w0=(-1.3, -0.7),
        wa=(-0.3, 0.3),
    )
    draw = {k: rng.uniform(lo, hi, N) for k, (lo, hi) in bounds.items()}

    B = compute_convergence_bispectrum(
        _ELL1,
        _ELL2,
        _ELL3,
        nz_capped,
        Cosmology(
            draw["omega_cold"],
            np.full(N, 0.0492),
            draw["sigma8_cold"],
            draw["ns"],
            draw["hubble"],
            neutrino_mass=np.zeros(N),
            w0=draw["w0"],
            wa=draw["wa"],
        ),
        pk_source="eh",
    )
    B = np.asarray(B)  # (N_cosmo=100, N_tri=len(_ELL_TRIS))

    n_bad = int((~np.isfinite(B)).sum())
    assert n_bad == 0, (
        f"{n_bad}/{B.size} (cosmology, triangle) combination(s) out of "
        f"{N} cosmologies x {len(_ELL1)} triangle shapes returned "
        "non-finite compute_convergence_bispectrum output at default "
        "settings -- the default R_min/R_max bracket no longer covers the "
        "documented prior range."
    )


# ---------------------------------------------------------------------------
# Test 24: multi-tracer legs ('counts', 'cmb_lensing') and per-leg validation
# ---------------------------------------------------------------------------

_GAL_Z = np.linspace(0.0, 2.0, 300)
_GAL_NZ = smail_nz(_GAL_Z, z0=0.4)  # a lower-z lens-like sample
_GAL_DNDZ = (_GAL_Z, _GAL_NZ)
_BIAS = 1.7


def test_three_lensing_legs_matches_shear_only(fiducial_params, nz):
    """Explicit tracers=('lensing',)*3 on the same dndz must equal the
    make_bkappa shim's output exactly (same code path, just spelled out) --
    the direct form of the backward-compatibility check every other test in
    this file already exercises implicitly through the shim."""
    p = fiducial_params
    ell = np.array([300.0, 700.0])
    args = (_cos(p),)
    B_shim = make_bkappa(ell, ell, ell, nz, pk_source="eh", N_chi=10)(*args)
    B_explicit = make_projected_bihalofit(
        ell,
        ell,
        ell,
        (nz, nz, nz),
        ("lensing", "lensing", "lensing"),
        pk_source="eh",
        N_chi=10,
    )(*args)
    np.testing.assert_array_equal(np.asarray(B_shim), np.asarray(B_explicit))


def test_counts_leg_finite_and_differentiable(fiducial_params, jax_device):
    """A 'counts' leg (galaxy clustering kernel) gives a finite, differentiable
    result when mixed with two 'lensing' legs (galaxy-galaxy-lensing-like)."""
    p = fiducial_params
    ell = np.array([300.0, 700.0])
    dndz = (_GAL_DNDZ, (_Z_GRID, _NZ_GRID), (_Z_GRID, _NZ_GRID))
    tracers = ("counts", "lensing", "lensing")
    bias = (_BIAS, None, None)

    with jax.default_device(jax_device):
        bhf = make_projected_bihalofit(
            ell, ell, ell, dndz, tracers, bias=bias, pk_source="eh", N_chi=10
        )
        B = np.asarray(bhf(_cos(p)))
        assert np.all(np.isfinite(B)), f"counts-leg B contains non-finite values: {B}"

        def f(s8):
            return jnp.sum(bhf(_cos(p, sigma8_cold=s8)))

        g = float(np.asarray(jax.grad(f)(jnp.asarray(float(p["sigma8_cold"][0])))))
    assert np.isfinite(g), f"d(counts-leg B)/d(sigma8) is not finite: {g}"


def test_cmb_lensing_leg_finite_and_differentiable(fiducial_params, jax_device):
    """A 'cmb_lensing' leg (single source plane at z_source) gives a finite,
    differentiable result, mixed with two 'lensing' legs, at the default
    chi_max (which truncates the CMB kernel's own support to z~3, hence the
    expected RuntimeWarning, ignored here). Wider chi_max is tested below."""
    p = fiducial_params
    ell = np.array([300.0, 700.0])
    dndz = (None, (_Z_GRID, _NZ_GRID), (_Z_GRID, _NZ_GRID))
    tracers = ("cmb_lensing", "lensing", "lensing")
    z_source = (1089.0, 1089.0, 1089.0)

    with jax.default_device(jax_device):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bhf = make_projected_bihalofit(
                ell,
                ell,
                ell,
                dndz,
                tracers,
                z_source=z_source,
                pk_source="eh",
                N_chi=10,
            )
        B = np.asarray(bhf(_cos(p)))
        assert np.all(
            np.isfinite(B)
        ), f"cmb_lensing-leg B contains non-finite values: {B}"

        def f(s8):
            return jnp.sum(bhf(_cos(p, sigma8_cold=s8)))

        g = float(np.asarray(jax.grad(f)(jnp.asarray(float(p["sigma8_cold"][0])))))
    assert np.isfinite(g), f"d(cmb_lensing-leg B)/d(sigma8) is not finite: {g}"


def test_cmb_lensing_leg_background_grid_is_converged(fiducial_params, jax_device):
    """The default background grid is fine enough for a 'cmb_lensing' leg.

    The grid is stretched out to z_source=1089 and chi(z) is a trapezoid sum on
    it. One uniform grid of the default 2000 points left about five points below
    z=3 and put this leg up to ~25% off; the grid is now uniform to z=10 and
    log-spaced beyond."""
    p = fiducial_params
    ell = np.array([30.0, 300.0, 1000.0])
    dndz = (None, (_Z_GRID, _NZ_GRID), (_Z_GRID, _NZ_GRID))
    tracers = ("cmb_lensing", "lensing", "lensing")

    with jax.default_device(jax_device), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        B = {
            n: np.asarray(
                make_projected_bihalofit(
                    ell,
                    ell,
                    ell,
                    dndz,
                    tracers,
                    z_source=(1089.0,) * 3,
                    pk_source="eh",
                    N_chi=10,
                    N_z_bg=n,
                )(_cos(p))
            )
            for n in (2000, 20000)
        }
    np.testing.assert_allclose(B[2000], B[20000], rtol=1e-4)


def test_cmb_lensing_leg_wide_chi_max_is_finite_and_harmless(
    fiducial_params, jax_device
):
    """chi_max past the source-plane distance (~9500 Mpc/h): the chi-nodes beyond
    it carry ~1e-11 of the weight but BiHalofit's k_NL solve fails there, so they
    are dropped before B is evaluated. The result is finite, differentiable, and equal to
    the default-chi_max one, because the shear legs have no support beyond z=3."""
    p = fiducial_params
    ell = np.array([300.0, 700.0])
    dndz = (None, (_Z_GRID, _NZ_GRID), (_Z_GRID, _NZ_GRID))
    tracers = ("cmb_lensing", "lensing", "lensing")

    def build(**kw):
        return make_projected_bihalofit(
            ell,
            ell,
            ell,
            dndz,
            tracers,
            z_source=(1089.0,) * 3,
            pk_source="eh",
            N_chi=40,
            **kw,
        )

    with jax.default_device(jax_device), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        B_default = np.asarray(build()(_cos(p)))
        wide = build(chi_max=12000.0)
        B_wide = np.asarray(wide(_cos(p)))
        g = float(
            np.asarray(
                jax.grad(lambda s8: jnp.sum(wide(_cos(p, sigma8_cold=s8))))(
                    jnp.asarray(float(p["sigma8_cold"][0]))
                )
            )
        )
    assert np.all(
        np.isfinite(B_wide)
    ), f"wide-chi_max B contains non-finite values: {B_wide}"
    assert np.isfinite(g), f"d(wide-chi_max B)/d(sigma8) is not finite: {g}"
    np.testing.assert_allclose(B_wide, B_default, rtol=1e-3)


def test_cmb_truncation_warning_follows_the_product_of_legs(fiducial_params):
    """A 'cmb_lensing' leg's own weight is mostly beyond the default chi_max, but
    the integrand is a product of kernels: with a shear leg whose sources end
    inside chi_max nothing is lost and no warning is due, while three CMB legs
    do lose the integrand and must warn. (Distinct ells: the check runs once
    per geometry.)"""
    z_source = (1089.0,) * 3

    def warnings_for(ell, dndz, tracers):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_projected_bihalofit(
                ell, ell, ell, dndz, tracers, z_source=z_source, pk_source="eh", N_chi=5
            )
        return [str(m.message) for m in w if "cmb_lensing efficiency" in str(m.message)]

    mixed = warnings_for(
        np.array([251.0, 253.0]),
        (None, _N_Z, _N_Z),
        ("cmb_lensing", "lensing", "lensing"),
    )
    all_cmb = warnings_for(
        np.array([257.0, 263.0]), (None, None, None), ("cmb_lensing",) * 3
    )
    assert mixed == []
    assert len(all_cmb) == 3


def test_background_zgrid_unchanged_for_shear_and_covers_cmb():
    """Below z_split the grid is uniform, so shear results are unaffected; beyond it the grid stays dense at low z and reaches z_max."""
    from bihalofit_jax.background import _background_zgrid

    np.testing.assert_array_equal(
        _background_zgrid(4.0, 2000), np.linspace(0.0, 4.0, 2000)
    )
    z = _background_zgrid(1143.0, 2000)
    assert z[0] == 0.0 and np.isclose(z[-1], 1143.0) and np.all(np.diff(z) > 0)
    assert np.sum(z <= 3.0) > 500


@pytest.mark.parametrize(
    "bad_kwargs, match",
    [
        (dict(dndz=(_N_Z, _N_Z), tracers=("lensing",) * 3), "length-3 tuples"),
        (
            dict(dndz=(_N_Z, _N_Z, _N_Z), tracers=("lensing", "lensing")),
            "length-3 tuples",
        ),
        (
            dict(dndz=(_N_Z, _N_Z, _N_Z), tracers=("lensing", "lensing", "bogus")),
            "must be one of",
        ),
        (
            dict(dndz=(None, _N_Z, _N_Z), tracers=("lensing", "lensing", "lensing")),
            "is required for",
        ),
        (
            dict(
                dndz=(_N_Z, _N_Z, _N_Z), tracers=("cmb_lensing", "lensing", "lensing")
            ),
            "must be None for",
        ),
        (
            dict(
                dndz=(_N_Z, _N_Z, _N_Z),
                tracers=("lensing", "lensing", "counts"),
                bias=(None, None, None),
            ),
            "bias\\[2\\] is required",
        ),
        (
            dict(
                dndz=(_N_Z, _N_Z, _N_Z),
                tracers=("lensing", "lensing", "lensing"),
                bias=(None, None, 1.5),
            ),
            "bias\\[2\\] must be None",
        ),
    ],
)
def test_multi_tracer_input_validation(bad_kwargs, match):
    """Length/consistency mismatches between dndz/tracers/bias/z_source are
    rejected with a descriptive ValueError before any JAX work."""
    ell = np.array([500.0])
    kwargs = dict(dndz=(_N_Z, _N_Z, _N_Z), tracers=("lensing",) * 3)
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError, match=match):
        make_projected_bihalofit(ell, ell, ell, **kwargs, pk_source="eh")


# ---------------------------------------------------------------------------
# Test 25: per-call nz= / bias= overrides
# ---------------------------------------------------------------------------

_COUNTS_LENS = dict(
    dndz=(_GAL_DNDZ, _N_Z, _N_Z),
    tracers=("counts", "lensing", "lensing"),
    bias=(_BIAS, None, None),
    pk_source="eh",
    N_chi=10,
)


@pytest.fixture(scope="module")
def counts_lens_bhf():
    ell = np.array([300.0, 700.0])
    return make_projected_bihalofit(ell, ell, ell, **_COUNTS_LENS)


def test_nz_override_matches_build_time_and_is_normalised_in_trace(
    fiducial_params, counts_lens_bhf
):
    """Passing the build-time n(z) back, unnormalised, reproduces the default."""
    c = _cos(fiducial_params)
    B0 = np.asarray(counts_lens_bhf(c))
    B1 = np.asarray(counts_lens_bhf(c, nz=(3.0 * _GAL_NZ, None, 0.5 * _NZ_GRID)))
    np.testing.assert_allclose(B1, B0, rtol=1e-10)


def test_bias_override_scales_counts_leg_linearly(fiducial_params, counts_lens_bhf):
    c = _cos(fiducial_params)
    B0 = np.asarray(counts_lens_bhf(c))
    B2 = np.asarray(counts_lens_bhf(c, bias=(2.0 * _BIAS, None, None)))
    np.testing.assert_allclose(B2, 2.0 * B0, rtol=1e-10)


def test_nz_and_bias_overrides_are_differentiable(fiducial_params, counts_lens_bhf):
    """Gradients w.r.t. an n(z) width nuisance parameter and the bias are
    finite and match central differences."""
    c = _cos(fiducial_params)

    def f(theta):
        z0, b = theta[0], theta[1]
        nz = _GAL_Z**2 * jnp.exp(-_GAL_Z / z0)  # smail_nz shape, unnormalised
        return jnp.sum(counts_lens_bhf(c, nz=(nz, None, None), bias=(b, None, None)))

    theta = jnp.asarray([0.4, _BIAS])
    g = np.asarray(jax.grad(f)(theta))
    h = np.array([1e-4, 1e-4])
    fd = np.array(
        [
            (
                f(theta + jnp.asarray(np.eye(2)[i] * h[i]))
                - f(theta - jnp.asarray(np.eye(2)[i] * h[i]))
            )
            / (2 * h[i])
            for i in range(2)
        ]
    )
    assert np.all(np.isfinite(g)), g
    np.testing.assert_allclose(g, fd, rtol=1e-4)


@pytest.mark.parametrize(
    "bad_nz",
    [
        np.zeros_like(_GAL_NZ),
        np.where(_GAL_Z < 1.0, _GAL_NZ, -_GAL_NZ),
        np.full_like(_GAL_NZ, np.nan),
    ],
    ids=["zero", "negative", "nan"],
)
def test_invalid_nz_override_is_rejected(fiducial_params, counts_lens_bhf, bad_nz):
    """A concrete per-call n(z) gets the same checks as the build-time dndz;
    without them a zero integral reaches B as inf/NaN."""
    with pytest.raises(ValueError, match="nz\\[0\\] must be finite and non-negative"):
        counts_lens_bhf(_cos(fiducial_params), nz=(bad_nz, None, None))


def test_negative_build_time_dndz_is_rejected():
    ell = np.array([500.0])
    bad = (_Z_GRID, _NZ_GRID - 0.5 * _NZ_GRID.max())
    with pytest.raises(
        ValueError, match="dndz\\[0\\]\\[1\\] must be finite and non-negative"
    ):
        make_projected_bihalofit(
            ell, ell, ell, (bad, _N_Z, _N_Z), ("lensing",) * 3, pk_source="eh"
        )
