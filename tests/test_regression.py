"""
pytest regression suite for bihalofit_jax.py.

Run full suite (all backends, all devices):
    pytest tests/test_regression.py -v

CPU only:
    pytest tests/test_regression.py -v --device cpu

GPU only:
    pytest tests/test_regression.py -v --device gpu

Limit to a single backend:
    pytest tests/test_regression.py -v --backend eh

CPU only, single backend:
    pytest tests/test_regression.py -v --device cpu --backend baccoemu

Run a single test:
    pytest tests/test_regression.py::test_output_shapes -v
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bihalofit_jax import Cosmology
from bihalofit_jax import make_bihalofit


def compute_bihalofit(expfactors, k1, k2, k3, cosmo=None, **kwargs):
    """Build and evaluate in one step."""
    return make_bihalofit(k1, k2, k3, expfactors, **kwargs)(cosmo)


# Monkeypatch targets must be the module where the name is *looked up*, not
# wherever it happens to be re-exported: _resolve_N_R resolves _tol_to_niter
# in _knl_solver's globals, and make_bihalofit's closure resolves
# bispectrum_all_z in bispectrum's globals.
import bihalofit_jax.bispectrum as _bhfk
import bihalofit_jax._knl_solver as _bhfs
from bihalofit_jax._knl_solver import _tol_to_niter, _get_solver, _solve_knl_neff
from bihalofit_jax._pk_backends import _pk_backend


# ---------------------------------------------------------------------------
# Generic helper: dispatch to any backend
# ---------------------------------------------------------------------------


def _call(backend, k1, k2, k3, expfactors, params, **kw):
    # params may be a dict of the Cosmology field names or a Cosmology;
    # the returned callable accepts either.
    return make_bihalofit(k1, k2, k3, expfactors, pk_source=backend, **kw)(params)


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


def _call_with_jac(backend, k1, k2, k3, expfactors, params, wrt, **kw):
    """Primal + Jacobian w.r.t. one or more named cosmological parameters.

    Cosmology is a pytree, so one jacfwd over the container returns a
    Cosmology of derivatives, selected by name rather than by argnums."""
    names = [wrt] if isinstance(wrt, str) else list(wrt)
    cosmo = params if isinstance(params, Cosmology) else Cosmology(**params)

    def _fwd(c):
        return _call(backend, k1, k2, k3, expfactors, c, **kw)

    B = _fwd(cosmo)
    jac = jax.jacfwd(_fwd)(cosmo)
    return B, {n: getattr(jac, n) for n in names}


# ---------------------------------------------------------------------------
# Test 1: golden reference agreement
# ---------------------------------------------------------------------------


def _golden_params(golden):
    """Load the cosmologies that were used when the golden file was generated."""
    return {
        k: golden[f"cosmo_{k}"]
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


def test_golden_reference(backend, jax_device, golden):
    """New run must agree with stored golden within 1%.

    Cosmologies are read from the golden file itself to prevent fixture/file drift.
    All backends can run on CPU and GPU.
    """
    p = _golden_params(golden)
    k1 = golden["k1"]
    k2 = golden["k2"]
    k3 = golden["k3"]
    a = 1.0 / (1.0 + golden["redshifts"])

    with jax.default_device(jax_device):
        B_new = np.asarray(_call(backend, k1, k2, k3, a, p))
    B_ref = golden[f"B_ref_{backend}"]
    ratio = B_new / B_ref
    max_dev = float(np.max(np.abs(ratio - 1.0)))
    assert (
        max_dev < 0.001
    ), f"{backend}: max |B/B_ref - 1| = {max_dev:.4f} exceeds 0.1% tolerance"


# ---------------------------------------------------------------------------
# Test 2: output shape contracts and finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N_cosmo,N_z,N_tri",
    [
        (1, 1, 10),
        (1, 4, 50),
        (3, 2, 100),
        (10, 1, 200),
    ],
)
def test_output_shapes(
    N_cosmo, N_z, N_tri, fiducial_params, triangles, redshifts, jax_device, backend
):
    """B.shape == (N_cosmo, N_z, N_tri) for various dimension combinations."""
    k1, k2, k3 = triangles
    k1 = k1[:N_tri]
    k2 = k2[:N_tri]
    k3 = k3[:N_tri]
    a = (1.0 / (1.0 + redshifts))[:N_z]
    p = {k: np.tile(v, N_cosmo) for k, v in fiducial_params.items()}

    with jax.default_device(jax_device):
        B = np.asarray(_call(backend, k1, k2, k3, a, p))

    assert B.shape == (
        N_cosmo,
        N_z,
        N_tri,
    ), f"Expected ({N_cosmo},{N_z},{N_tri}), got {B.shape}"


# Explicit representative triangles chosen to cover distinct physical regimes:
# equilateral at low/mid/high k, squeezed, near-degenerate (folded), and scalene.
# These are intentional, not a grid artefact, so the test is stable under
# any future change to the grid resolution or k range.
_FINITENESS_TRIANGLES = np.array(
    [
        [0.02, 0.02, 0.02],  # equilateral, linear regime
        [0.10, 0.10, 0.10],  # equilateral, transition
        [0.50, 0.50, 0.50],  # equilateral, deep nonlinear
        [0.10, 0.10, 0.02],  # squeezed (k3 << k1 ~ k2)
        [0.50, 0.50, 0.05],  # squeezed, high-k legs
        [0.10, 0.10, 0.195],  # near-degenerate (k3 ~ k1+k2)
        [0.30, 0.20, 0.48],  # near-degenerate, scalene
        [0.05, 0.15, 0.18],  # scalene, mid-k
        [0.01, 0.50, 0.50],  # extreme dynamic range
    ]
)


def test_output_finite_positive(fiducial_params, redshifts, jax_device, backend):
    """B is finite and positive for representative triangle configurations.

    Uses an explicit set covering equilateral, squeezed, near-degenerate, and
    scalene shapes across a wide k range, independent of any grid fixture.
    """
    k1 = _FINITENESS_TRIANGLES[:, 0]
    k2 = _FINITENESS_TRIANGLES[:, 1]
    k3 = _FINITENESS_TRIANGLES[:, 2]
    a = 1.0 / (1.0 + redshifts)

    with jax.default_device(jax_device):
        B = np.asarray(_call(backend, k1, k2, k3, a, fiducial_params))

    assert np.all(np.isfinite(B)), (
        f"{backend}: B contains non-finite values\n"
        f"  triangles: {_FINITENESS_TRIANGLES[~np.all(np.isfinite(B), axis=(0,1))]}"
    )
    assert np.all(B > 0), (
        f"{backend}: B contains non-positive values\n"
        f"  triangles: {_FINITENESS_TRIANGLES[~np.all(B > 0, axis=(0,1))]}"
    )


# ---------------------------------------------------------------------------
# Test 3: Jacobian shape
# ---------------------------------------------------------------------------


def test_jacobian_shape(fiducial_params, triangles, redshifts, jax_device, backend):
    """Jacobian wrt sigma8_cold has shape (N_c, N_z, N_tri, N_c) and is finite."""
    N_c, N_z, N_tri = 3, 2, 100
    k1, k2, k3 = triangles
    k1 = k1[:N_tri]
    k2 = k2[:N_tri]
    k3 = k3[:N_tri]
    a = (1.0 / (1.0 + redshifts))[:N_z]
    p = {k: np.tile(v, N_c) for k, v in fiducial_params.items()}

    with jax.default_device(jax_device):
        _, jac = _call_with_jac(backend, k1, k2, k3, a, p, "sigma8_cold")
    J = np.asarray(jac["sigma8_cold"])

    assert J.shape == (
        N_c,
        N_z,
        N_tri,
        N_c,
    ), f"Expected ({N_c},{N_z},{N_tri},{N_c}), got {J.shape}"
    assert np.all(np.isfinite(J)), "Jacobian contains non-finite values"


# ---------------------------------------------------------------------------
# Test 4: float32 / float64 agreement
# ---------------------------------------------------------------------------


def test_dtype_agreement(
    fiducial_params, test_triangles, redshifts, jax_device, backend
):
    """float32 and float64 results agree to within 0.1%."""
    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)

    with jax.default_device(jax_device):
        B64 = np.asarray(
            _call(backend, k1, k2, k3, a, fiducial_params, dtype=np.float64)
        )
        B32 = np.asarray(
            _call(backend, k1, k2, k3, a, fiducial_params, dtype=np.float32)
        )

    max_reldiff = float(np.max(np.abs(B32 - B64) / np.abs(B64)))
    assert (
        max_reldiff < 1e-3
    ), f"{backend}: float32/float64 max relative difference {max_reldiff:.2e} exceeds 1e-3"


def test_syren_float32_gradients_at_zero_neutrino_mass(test_triangles):
    """float32 syren gradients at neutrino_mass=0 (the Cosmology default) are
    finite and match float64. They were NaN before _growth_D_syren's
    free-streaming term was moved to log space."""
    k1, k2, k3 = test_triangles
    a = np.array([1.0, 0.5])
    cosmo = Cosmology(
        omega_cold=0.3156,
        omega_baryon=0.0492,
        sigma8_cold=0.831,
        ns=0.9645,
        hubble=0.6727,
        neutrino_mass=0.0,
        w0=-0.95,
        wa=0.1,
    )
    grads = {}
    for dt in (np.float32, np.float64):
        bhf = make_bihalofit(k1, k2, k3, a, pk_source="syren", dtype=dt)
        g = jax.grad(lambda c: jax.numpy.log(bhf(c)).sum())(cosmo)
        grads[dt] = np.array([float(getattr(g, n)) for n in _COSMO_NAMES])
    assert np.all(np.isfinite(grads[np.float32])), dict(
        zip(_COSMO_NAMES, grads[np.float32])
    )
    np.testing.assert_allclose(grads[np.float32], grads[np.float64], rtol=1e-3)


# ---------------------------------------------------------------------------
# Test 5: permutation invariance
# ---------------------------------------------------------------------------


def test_permutation_invariance(
    fiducial_params, test_triangles, redshifts, jax_device, backend
):
    """B(k1,k2,k3) == B(k2,k1,k3) == B(k3,k2,k1) to within a few ULPs."""
    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)

    with jax.default_device(jax_device):
        B123 = np.asarray(_call(backend, k1, k2, k3, a, fiducial_params))
        B213 = np.asarray(_call(backend, k2, k1, k3, a, fiducial_params))
        B321 = np.asarray(_call(backend, k3, k2, k1, a, fiducial_params))

    # Cyclic sum reorders floating-point additions across permutations;
    # differences are ≤ a few ULPs of float64 (rtol=1e-14 ≈ 50× eps).
    rtol = 1e-5 if backend == "baccoemu" else 1e-14
    np.testing.assert_allclose(
        B123,
        B213,
        rtol=rtol,
        err_msg=f"{backend}: B(k1,k2,k3) != B(k2,k1,k3): permutation symmetry broken",
    )
    np.testing.assert_allclose(
        B123,
        B321,
        rtol=rtol,
        err_msg=f"{backend}: B(k1,k2,k3) != B(k3,k2,k1): permutation symmetry broken",
    )


# ---------------------------------------------------------------------------
# Test 6: primal / Jacobian consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrt",
    [
        "omega_cold",
        "omega_baryon",
        "sigma8_cold",
        "ns",
        "hubble",
        "neutrino_mass",
        "w0",
        "wa",
        pytest.param(["sigma8_cold", "ns"], id="sigma8+ns"),
        pytest.param(["omega_cold", "hubble", "sigma8_cold"], id="om+h+s8"),
    ],
)
def test_primal_jacobian_consistency(
    fiducial_params, test_triangles, redshifts, jax_device, backend, wrt
):
    """Primal B returned alongside any Jacobian is bit-for-bit identical to the primal-only call.

    Covers every individual cosmological parameter and two multi-parameter combinations,
    guarding against branches or caching bugs that corrupt the primal under jacfwd.
    """
    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)

    with jax.default_device(jax_device):
        B_primal = np.asarray(_call(backend, k1, k2, k3, a, fiducial_params))
        B_from_jac = np.asarray(
            _call_with_jac(backend, k1, k2, k3, a, fiducial_params, wrt)[0]
        )

    np.testing.assert_array_equal(
        B_primal,
        B_from_jac,
        err_msg=f"{backend} wrt={wrt}: primal B and wrt-path primal disagree",
    )


# ---------------------------------------------------------------------------
# Test 7: finite-difference vs AD Jacobian
# ---------------------------------------------------------------------------


def test_fd_vs_ad_jacobian(fiducial_params, redshifts, jax_device, backend):
    """AD dB/domega_cold agrees with central FD over the full triangle grid.

    Two things decide whether this comparison is meaningful, and both bite:

    * **The FD step cannot be tiny.** Central differences balance truncation
      (~h^2) against the noise floor of B divided by h. For 'eh'/'syren', B is an
      analytic formula in float64 and that floor is ~1e-16, but 'baccoemu' is a
      neural network with float32 weights, so its floor is ~1e-7. A step of 1e-7
      then makes the FD estimate pure noise -- measured: 213% error at that step,
      against 3e-5 at a relative step of 1e-3. The step is therefore relative to
      the parameter, and large enough for the emulator.
    * **Relative error is meaningless where the derivative vanishes.** At high k
      some triangles carry |dB/dOm| ~ 1e-7 of the largest in the set, so the
      denominator is floored at 1e-6 of the largest |FD| -- an absolute tolerance
      on a scale set by the data, rather than a relative one against ~0.
    """

    def log_k_triangles(kmin=0.01, kmax=1.0, dlogk=0.1):
        """Every closed triangle on a log-spaced k grid, k1 <= k2 <= k3."""
        k = np.exp(np.arange(np.log(kmin), np.log(kmax) + dlogk / 2, dlogk))
        tri = np.array(
            [
                (ki, kj, kk)
                for i, ki in enumerate(k)
                for j, kj in enumerate(k[i:], i)
                for kk in k[j:]
                if kk <= ki + kj
            ]
        )
        return tri[:, 0], tri[:, 1], tri[:, 2]

    k1_t, k2_t, k3_t = log_k_triangles()
    a = np.array([1.0])  # z = 0
    om = float(fiducial_params["omega_cold"][0])
    h = 1e-3 * om  # relative step; see the docstring

    def _eval(omega_cold):
        p = dict(fiducial_params)
        p["omega_cold"] = np.array([omega_cold])
        with jax.default_device(jax_device):
            return np.asarray(_call(backend, k1_t, k2_t, k3_t, a, p), dtype=np.float64)[
                0, 0
            ]

    dB_FD = (_eval(om + h) - _eval(om - h)) / (2.0 * h)
    with jax.default_device(jax_device):
        _, jac = _call_with_jac(
            backend, k1_t, k2_t, k3_t, a, fiducial_params, "omega_cold"
        )
    dB_AD = np.asarray(jac["omega_cold"], dtype=np.float64)[0, 0, :, 0]

    rel = np.abs(dB_AD - dB_FD) / np.maximum(np.abs(dB_FD), np.abs(dB_FD).max() * 1e-6)
    i = int(np.argmax(rel))
    assert rel.max() < 0.01, (
        f"{backend}: max AD vs FD error {rel.max():.2e} over {len(k1_t)} triangles "
        f"(median {np.median(rel):.1e}); worst at k=({k1_t[i]:.3f}, {k2_t[i]:.3f}, "
        f"{k3_t[i]:.3f}) where AD={dB_AD[i]:.4e} and FD={dB_FD[i]:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 7b: gradient continuity across a k_NL grid crossing
# ---------------------------------------------------------------------------


def test_gradient_continuity_across_knl_crossing(fiducial_params, jax_device):
    """dB/dsigma8 must vary smoothly as sigma8 sweeps across k_NL R-grid crossings.

    k_NL solves sigma^2(R_NL)=1 on a fixed R grid. sweeping sigma8 moves the
    crossing continuously across many grid cells over this range (N_R=500
    default), repeatedly exercising whatever bracket/interpolation logic
    locates it. A discontinuous bracket (e.g. argmax + linear interp, whose
    selected grid index jumps discretely) kinks dB/dsigma8 at every crossing;
    the curvature (second difference) of the AD gradient spikes by orders of
    magnitude at those kinks relative to its typical smooth variation. A C^1
    bracket keeps that spike within normal smooth-curve variation.

    Only tested against pk_source='eh': the bracket/root-find logic that
    locates k_NL is shared by all three backends (only the input P(k) differs),
    so this is a property of that shared code path, not of any one backend.
    """
    k1 = np.array([0.10])
    k2 = np.array([0.15])
    k3 = np.array([0.20])
    a = np.array([1.0])  # z=0

    s8_grid = np.linspace(0.70, 0.95, 400)
    p = dict(fiducial_params)

    grads = np.empty_like(s8_grid)
    with jax.default_device(jax_device):
        for i, s8 in enumerate(s8_grid):
            p["sigma8_cold"] = np.array([s8])
            _, jac = _call_with_jac("eh", k1, k2, k3, a, p, "sigma8_cold")
            grads[i] = float(np.asarray(jac["sigma8_cold"])[0, 0, 0, 0])

    assert np.all(np.isfinite(grads)), "dB/dsigma8 is non-finite somewhere on the sweep"

    d2 = np.diff(grads, n=2)
    ratio = float(np.max(np.abs(d2)) / np.median(np.abs(d2)))
    assert ratio < 100.0, (
        f"gradient curvature spikes by {ratio:.1f}x the typical smooth "
        f"variation over the sigma8 sweep -- looks like a discontinuous "
        f"kink at a k_NL grid crossing, not a C^1 bracket"
    )


def test_gradient_continuity_across_alphan_min(fiducial_params, jax_device):
    """dB/dns must vary smoothly across the alphan = min(f(neff), g(ns)) branch swap.

    alphan is defined as the min of two fitting-formula branches; sweeping ns
    moves g(ns) = 1 - 2/3*ns continuously across f(neff), repeatedly crossing
    which branch is selected. A hard jnp.minimum kinks dB/dns at every
    crossing; the curvature (second difference) of the AD gradient spikes by
    orders of magnitude at those kinks relative to its typical smooth
    variation. A smooth (soft) min keeps that spike within normal
    smooth-curve variation.

    Squeezed triangle (r2 far from 1) so alphan's neff/r2 branch varies
    strongly enough over the ns sweep to actually cross the ns branch.
    """
    k1 = np.array([0.30])
    k2 = np.array([0.25])
    k3 = np.array([0.45])
    a = np.array([1.0])  # z=0

    ns_grid = np.linspace(0.80, 1.10, 400)
    p = dict(fiducial_params)

    grads = np.empty_like(ns_grid)
    with jax.default_device(jax_device):
        for i, ns_ in enumerate(ns_grid):
            p["ns"] = np.array([ns_])
            _, jac = _call_with_jac("eh", k1, k2, k3, a, p, "ns")
            grads[i] = float(np.asarray(jac["ns"])[0, 0, 0, 0])

    assert np.all(np.isfinite(grads)), "dB/dns is non-finite somewhere on the sweep"

    d2 = np.diff(grads, n=2)
    ratio = float(np.max(np.abs(d2)) / np.median(np.abs(d2)))
    assert ratio < 100.0, (
        f"gradient curvature spikes by {ratio:.1f}x the typical smooth "
        f"variation over the ns sweep -- looks like a discontinuous kink at "
        f"an alphan branch swap, not a smooth min"
    )


@pytest.mark.slow
def test_gradient_continuity_random_directions(jax_device):
    """dB/dt must vary smoothly along random walks through the full 8-param space.

    The targeted continuity tests above each sweep one parameter through one
    known kink site (k_NL crossing, alphan branch swap). This test doesn't
    target a specific mechanism: it walks a fixed, seeded set of random
    directions through the full cosmology vector and applies the same
    curvature-spike check anywhere it happens to cross a kink, whatever the
    cause. It's a coarse net, not a replacement for the targeted tests --
    a direction can miss a kink entirely (this test's worst-case ratio is
    ~22 on the current C^1 implementation, versus ~4500 on the original
    unsmoothed code -- but a single unlucky direction on unsmoothed code
    still came in at only ~11, so this test alone would not reliably catch
    a regression). Keep both: targeted tests for sensitivity and a
    pinpointable failure, this one as a generic guard against kinks
    introduced anywhere else in the pipeline.
    """
    k1 = np.array([0.10, 0.30])
    k2 = np.array([0.15, 0.25])
    k3 = np.array([0.20, 0.45])
    a = np.array([1.0, 0.7])

    names = [
        "omega_cold",
        "omega_baryon",
        "sigma8_cold",
        "ns",
        "hubble",
        "neutrino_mass",
        "w0",
        "wa",
    ]
    theta0 = jnp.array([0.3156, 0.0492, 0.831, 0.9645, 0.6727, 0.0, -1.0, 0.0])
    # per-parameter scale keeping the random line inside a sane prior box
    scale = jnp.array([0.08, 0.01, 0.12, 0.08, 0.10, 0.05, 0.1, 0.15])

    def g(t, direction):
        theta = theta0 + t * direction
        kw = {n: jnp.atleast_1d(theta[i]) for i, n in enumerate(names)}
        B = compute_bihalofit(a, k1, k2, k3, Cosmology(**kw), pk_source="eh")
        return jnp.sum(B)

    rng = np.random.default_rng(0)  # fixed seed: reproducible, not flaky
    t_grid = np.linspace(-1.0, 1.0, 250)
    worst_ratio = 0.0

    with jax.default_device(jax_device):
        for _ in range(12):
            direction = jnp.asarray(rng.normal(size=8) * np.asarray(scale))
            grads = np.array(
                [float(jax.grad(g)(jnp.asarray(t), direction)) for t in t_grid]
            )
            assert np.all(
                np.isfinite(grads)
            ), "dB/dt is non-finite somewhere along a random direction"
            d2 = np.diff(grads, n=2)
            worst_ratio = max(
                worst_ratio, float(np.max(np.abs(d2)) / np.median(np.abs(d2)))
            )

    assert worst_ratio < 100.0, (
        f"gradient curvature spikes by {worst_ratio:.1f}x the typical smooth "
        f"variation along a random direction through parameter space -- "
        f"looks like a discontinuous kink somewhere in the pipeline"
    )


# ---------------------------------------------------------------------------
# Test 8: σ²=1 guard
# ---------------------------------------------------------------------------


def test_sigma2_guard(fiducial_params, test_triangles, redshifts, jax_device, backend):
    """Very small sigma8 (k_NL undefined): NaN by default, ValueError if validate=True."""
    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)
    p = dict(fiducial_params)
    p["sigma8_cold"] = np.array([0.01])  # guaranteed never to cross σ²=1

    with jax.default_device(jax_device):
        # Default: kernel poisons the invalid entries with NaN (no raise).
        B = np.asarray(_call(backend, k1, k2, k3, a, p))
        assert np.all(
            np.isnan(B)
        ), f"{backend}: expected all-NaN for σ²<1 everywhere, got finite values"

        # Opt-in: validate=True raises a descriptive error.
        with pytest.raises(ValueError, match=r"sigma\^2.*never reaches 1"):
            _call(backend, k1, k2, k3, a, p, validate=True)


def test_sigma2_guard_per_element(
    fiducial_params, test_triangles, redshifts, jax_device, backend
):
    """In a batch, only the out-of-domain cosmology is NaN; valid ones stay finite.

    A single bad cosmology in a vectorised MCMC step must not kill the whole batch.
    """
    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)
    p2 = {k: np.array([float(v[0]), float(v[0])]) for k, v in fiducial_params.items()}
    p2["sigma8_cold"] = np.array([float(fiducial_params["sigma8_cold"][0]), 0.01])

    with jax.default_device(jax_device):
        B = np.asarray(_call(backend, k1, k2, k3, a, p2))  # (2, N_z, N_tri)

    assert np.all(np.isfinite(B[0])), f"{backend}: valid cosmology became NaN"
    assert np.all(np.isnan(B[1])), f"{backend}: invalid cosmology not poisoned to NaN"


# ---------------------------------------------------------------------------
# Test 9: vectorisation consistency
# ---------------------------------------------------------------------------


def test_vectorisation_consistency(
    golden_cosmo_params, test_triangles, redshifts, jax_device, backend
):
    """Batched N_cosmo=5 agrees with stacked N_cosmo=1 × 5 runs.

    eh/syren: pure JAX float64; rtol=1e-12 (5000× eps).
    baccoemu: TF neural network with float32 weights; batch-size changes alter
    matmul summation order, so differences at the ~1e-6 level are expected.
    rtol=1e-5 catches real vectorisation bugs while accepting float32 rounding.
    """
    # Per-backend tolerance reflecting internal precision
    rtol = 1e-5 if backend == "baccoemu" else 1e-12

    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)
    N = 5

    with jax.default_device(jax_device):
        B_batch = np.asarray(
            _call(
                backend,
                k1,
                k2,
                k3,
                a,
                {k: v[:N] for k, v in golden_cosmo_params.items()},
            )
        )
        slices = [
            np.asarray(
                _call(
                    backend,
                    k1,
                    k2,
                    k3,
                    a,
                    {k: v[i : i + 1] for k, v in golden_cosmo_params.items()},
                )
            )
            for i in range(N)
        ]
    B_stacked = np.concatenate(slices, axis=0)

    np.testing.assert_allclose(
        B_batch,
        B_stacked,
        rtol=rtol,
        err_msg=f"{backend}: batched and per-cosmology results differ",
    )


# ---------------------------------------------------------------------------
# Test 10: CPU / GPU agreement
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_cpu_gpu_agreement(
    fiducial_params, test_triangles, redshifts, cpu_device, gpu_device, backend
):
    """CPU and GPU produce identical results to within device-rounding tolerance.

    eh/syren: pure JAX float64; rtol=1e-10 (~500 000× float64 eps).
    baccoemu: TF float32 NN; CPU and GPU use different float32 implementations,
    so ~1e-6 differences are expected. rtol=1e-5 catches real divergences.
    """
    rtol = 1e-5 if backend == "baccoemu" else 1e-10

    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)

    with jax.default_device(cpu_device):
        B_cpu = np.asarray(_call(backend, k1, k2, k3, a, fiducial_params))

    with jax.default_device(gpu_device):
        B_gpu = np.asarray(_call(backend, k1, k2, k3, a, fiducial_params))

    np.testing.assert_allclose(
        B_cpu, B_gpu, rtol=rtol, err_msg=f"{backend}: CPU and GPU results disagree"
    )


# ---------------------------------------------------------------------------
# Test 11: backend cross-agreement
# ---------------------------------------------------------------------------


def test_backend_cross_agreement(
    fiducial_params, test_triangles, redshifts, jax_device
):
    """baccoemu vs syren: max|ratio-1| < 1%; baccoemu vs E&H: < 30%."""
    pytest.importorskip("baccoemu")
    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + np.array([0.0]))  # z=0 only for speed

    with jax.default_device(jax_device):
        B_bemu = np.asarray(_call("baccoemu", k1, k2, k3, a, fiducial_params))
        B_syren = np.asarray(_call("syren", k1, k2, k3, a, fiducial_params))
        B_eh = np.asarray(_call("eh", k1, k2, k3, a, fiducial_params))

    max_bs = float(np.max(np.abs(B_bemu / B_syren - 1.0)))
    max_be = float(np.max(np.abs(B_bemu / B_eh - 1.0)))

    assert max_bs < 0.01, f"baccoemu vs syren: max|ratio-1| = {max_bs:.4f} exceeds 1%"
    assert max_be < 0.3, f"baccoemu vs E&H: max|ratio-1| = {max_be:.4f} exceeds 30%"


# ---------------------------------------------------------------------------
# Test 12: Takahashi C code agreement
# ---------------------------------------------------------------------------

# Measured worst cases over 3447 triangles, z <= 3, w0 in {-1, -0.9}: eh 0.04%,
# syren 0.4%, baccoemu 2.4% (baccoemu's P_lin(k, a) at k ~ 1 h/Mpc, z > 0,
# departs from D(z)^2 P(k, 0) in wCDM; its growth itself agrees to 0.1%).
_TAKAHASHI_TOL = {"eh": 0.01, "syren": 0.01, "baccoemu": 0.03}


def test_against_takahashi_reference(backend, jax_device, takahashi_reference):
    """Python BiHaloFit agrees with the Takahashi C code, for LCDM and w0=-0.9.

    The C code is fed the same z=0 P(k) that our Python backend produces and
    solves the growth ODE itself, so a discrepancy reflects the BiHaloFit
    formula integration (k_NL / n_eff root-finding, R-grid density) or the
    growth factor -- the w0 != -1 cosmology is what checks the latter.
    """
    ref = takahashi_reference
    a = 1.0 / (1.0 + ref["redshifts"])
    params = {k: ref[k] for k in _COSMO_NAMES}

    with jax.default_device(jax_device):
        B_py = np.asarray(_call(backend, ref["k1"], ref["k2"], ref["k3"], a, params))
    B_ref = ref[f"B_c_{backend}"]  # (N_cosmo, N_z, N_tri)

    for ic, w0 in enumerate(params["w0"]):
        finite = np.isfinite(B_ref[ic]) & np.isfinite(B_py[ic])
        max_dev = float(np.max(np.abs(B_py[ic][finite] / B_ref[ic][finite] - 1.0)))
        assert max_dev < _TAKAHASHI_TOL[backend], (
            f"{backend}, w0={w0}: max |B_py/B_c - 1| = {max_dev:.4f} "
            f"exceeds {_TAKAHASHI_TOL[backend]:.0%}"
        )


def test_baryonic_boost_against_takahashi_reference(takahashi_reference):
    """_baryon_ratio must match the Takahashi C code's own baryon_ratio()
    (App. C, Eq. C1) directly -- unlike test_baryonic_boost_matches_reference,
    which only checks internal self-consistency against a second Python
    transcription of the same formula. baryon_ratio has no P(k)/cosmology
    dependence (pure function of k1,k2,k3,z), so one comparison covers every
    backend. Skips automatically when the reference file predates this field.
    """
    ref = takahashi_reference
    if "Rb_c" not in ref.files:
        pytest.skip("Takahashi reference file predates Rb_c")
    k1, k2, k3 = ref["k1"], ref["k2"], ref["k3"]
    Rb_c = ref["Rb_c"]  # (N_z, N_tri)

    for iz, z in enumerate(ref["redshifts"]):
        a = 1.0 / (1.0 + z)
        Rb_py = np.asarray(_bhfk._baryon_ratio(k1, k2, k3, a))
        np.testing.assert_allclose(
            Rb_py,
            Rb_c[iz],
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"baryon_ratio mismatch at z={z}",
        )


# ---------------------------------------------------------------------------
# Error handling and edge cases
# ---------------------------------------------------------------------------

_K1 = np.array([0.1])
_A0 = np.array([1.0])


def test_invalid_pk_source(fiducial_params):
    """Unrecognised pk_source raises ValueError before any JAX work."""
    with pytest.raises(ValueError, match="pk_source must be one of"):
        compute_bihalofit(_A0, _K1, _K1, _K1, pk_source="bogus", cosmo=fiducial_params)


def test_debug_logging_does_not_crash(fiducial_params, caplog):
    """The kernel's debug logging is reachable and harmless.

    `verbose=` was a compute_bihalofit argument; that wrapper is gone, so
    callers set the level on the module logger themselves."""
    import logging

    with caplog.at_level(logging.DEBUG, logger="bihalofit_jax.bispectrum"):
        B = make_bihalofit(_K1, _K1, _K1, _A0, pk_source="eh")(fiducial_params)
    assert np.all(np.isfinite(np.asarray(B)))


def test_missing_hubble_raises():
    """All backends require hubble; omitting it raises ValueError."""
    params = dict(
        omega_cold=np.array([0.3156]),
        omega_baryon=np.array([0.0492]),
        sigma8_cold=np.array([0.831]),
        ns=np.array([0.9645]),
        neutrino_mass=np.array([0.0]),
        w0=np.array([-1.0]),
        wa=np.array([0.0]),
    )
    for backend in ("eh", "syren", "baccoemu"):
        # hubble is a required Cosmology field, so this now fails at container
        # construction rather than inside the kernel.
        with pytest.raises(TypeError, match="hubble"):
            compute_bihalofit(_A0, _K1, _K1, _K1, cosmo=params, pk_source=backend)


def test_neutrino_mass_none_uses_default(fiducial_params):
    """Explicitly passing neutrino_mass=None silently falls back to 0."""
    params = {k: v for k, v in fiducial_params.items() if k != "neutrino_mass"}
    B = compute_bihalofit(
        _A0, _K1, _K1, _K1, pk_source="eh", cosmo={**params, "neutrino_mass": None}
    )
    assert np.all(np.isfinite(np.asarray(B)))


def test_baccoemu_out_of_bounds_raises(fiducial_params):
    """Parameters outside the baccoemu trained prior raise instead of
    silently extrapolating the network (direct model access bypasses
    baccoemu's own validation)."""
    pytest.importorskip("baccoemu")
    p = dict(fiducial_params)
    p["omega_cold"] = np.array([0.9])  # baccoemu bounds: [0.06, 0.7]
    with pytest.raises(ValueError, match="baccoemu emulator bounds"):
        compute_bihalofit(_A0, _K1, _K1, _K1, cosmo=p, pk_source="baccoemu")


# ---------------------------------------------------------------------------
# tol / N_R / validate: the Newton-iteration-count knob, its tol-driven
# resolution (_tol_to_niter), and the unconditional verify-and-retry
# mechanism gated by validate.
# ---------------------------------------------------------------------------


def test_tol_N_R_mutually_exclusive(fiducial_params):
    """Passing both tol and a non-default N_R raises ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_bihalofit(
            _A0, _K1, _K1, _K1, pk_source="eh", tol=1e-8, N_R=5, cosmo=fiducial_params
        )


def test_tol_matches_manual_N_R_when_sufficient(fiducial_params):
    """tol=1e-8 resolves to N_R (see _tol_to_niter); the two calls must
    agree bit-for-bit. This only holds because _tol_to_niter pads its
    result 2 iterations past the point of convergence, not merely to the
    minimal-sufficient count -- AT the minimal-sufficient N_R, the Newton
    iteration is still an active, floating-point-path-sensitive
    computation, and a tol-resolved call vs. a manually-N_R-specified call
    at that exact N_R can differ at the ULP level (~1e-8 to 1e-14 relative,
    measured before the padding was added) because the tol path's residual
    check/retry apparatus wraps the core kernel in a slightly different
    compiled graph. Two-plus iterations past convergence, R has reached its
    exact float64 fixed point and every further iteration is a
    bit-identical no-op regardless of which call path produced it -- see
    _tol_to_niter's docstring."""
    B_tol = compute_bihalofit(
        _A0, _K1, _K1, _K1, pk_source="eh", tol=1e-8, cosmo=fiducial_params
    )
    B_N_R = compute_bihalofit(
        _A0,
        _K1,
        _K1,
        _K1,
        pk_source="eh",
        N_R=_tol_to_niter(1e-8),
        cosmo=fiducial_params,
    )
    np.testing.assert_array_equal(np.asarray(B_tol), np.asarray(B_N_R))


def test_tol_to_niter_monotonic():
    """A tighter (smaller) tol never resolves to fewer Newton iterations,
    and the result is always capped at 30."""
    tols = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-20]
    n_iters = [_tol_to_niter(t) for t in tols]
    assert all(
        n2 >= n1 for n1, n2 in zip(n_iters, n_iters[1:])
    ), f"non-monotonic n_iter sequence: {list(zip(tols, n_iters))}"
    assert n_iters[-1] <= 30


def test_tol_to_niter_meets_claimed_accuracy():
    """_tol_to_niter's resulting N_R actually achieves the requested k_NL
    tolerance, even for stressed (non-fiducial) cosmologies.

    Regression test for a bug where the loose-tolerance early-exit
    (tol >= E_WARMUP) skipped the +2 safety margin applied everywhere else:
    at tol=1e-2 under extreme cosmologies (sigma8/w0/wa away from LCDM) the
    actual error reached 0.147 (14.7x over the requested tolerance) before
    the fix. Measures the true k_NL error directly against a 30-iteration
    reference, independent of the B(k) fitting-formula amplification the
    original bug was found through.
    """
    rng = np.random.default_rng(0)
    N = 100
    oc = rng.uniform(0.15, 0.5, N)
    ob = np.full(N, 0.0492)
    h_ = rng.uniform(0.55, 0.85, N)
    ns_ = rng.uniform(0.85, 1.05, N)
    s8 = rng.uniform(0.4, 1.3, N)
    mnu = np.zeros(N)
    w0_ = rng.uniform(-1.3, -0.7, N)
    wa_ = rng.uniform(-0.3, 0.3, N)
    a = jnp.array([1.0])

    pk_fn = _pk_backend("eh")
    _, PL = pk_fn(
        jnp.asarray(oc),
        jnp.asarray(ob),
        jnp.asarray(h_),
        jnp.asarray(ns_),
        jnp.asarray(s8),
        jnp.asarray(mnu),
        jnp.asarray(w0_),
        jnp.asarray(wa_),
        a,
        jnp.float64,
    )
    PL_flat = PL.reshape(-1, PL.shape[-1])

    k2, k4, dk_over_2pi2, _ = _get_solver()
    R_min, R_max = 0.05, 40.0
    knl_ref, _, crossed_ref, _ = _solve_knl_neff(
        PL_flat, k2, k4, dk_over_2pi2, R_min, R_max, 30
    )
    knl_ref, crossed_ref = np.asarray(knl_ref), np.asarray(crossed_ref)

    for tol in [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
        n_iter = _tol_to_niter(tol)
        knl, _, crossed, _ = _solve_knl_neff(
            PL_flat, k2, k4, dk_over_2pi2, R_min, R_max, n_iter
        )
        knl, crossed = np.asarray(knl), np.asarray(crossed)
        mask = crossed_ref & crossed  # exclude bracket failures (not a tol problem)
        err = np.abs(knl[mask] / knl_ref[mask] - 1.0)
        assert err.max() <= tol, (
            f"tol={tol:g} (n_iter={n_iter}) violated: max k_NL error "
            f"{err.max():.3e} across {mask.sum()} stressed cosmologies"
        )


def test_N_R_alone_no_verification(fiducial_params):
    """Using N_R directly (no tol) never triggers the residual check, even
    for a deliberately too-small N_R and validate=True: only the bracket
    check applies when tol is None, so a merely-inaccurate (not undefined)
    k_NL still returns a finite result.

    N_R=2 here, not 1: with R_min widened to 1e-6 (was 0.05) for extreme
    low-h/high-z cosmologies, a single Newton step from anywhere near that
    much more distant bracket edge can land on a wildly wrong n_eff that
    overflows the fitting formula's power-law terms -- a genuine numerical
    limit of one iteration against a much wider bracket, not an undefined
    root (the bracket-crossing check this test targets is unaffected; N_R=2
    already recovers a finite, still deliberately unconverged, result)."""
    B = compute_bihalofit(
        _A0, _K1, _K1, _K1, pk_source="eh", N_R=2, validate=True, cosmo=fiducial_params
    )
    assert np.all(np.isfinite(np.asarray(B)))


def test_tol_under_estimate_raises_and_names_N_R(monkeypatch, fiducial_params):
    """If _tol_to_niter under-estimates the iteration count, validate=True
    reports it rather than silently returning an under-converged value.

    There is no automatic re-solve: tol only sizes the iteration count up
    front, so the error names N_R as the fix. (Before, this path doubled N_R
    and retried, which cost the callable its jittability for a case that
    needs the estimator sabotaged to reach at all.)"""
    _k1 = np.array([0.114])  # own triangle side, never reused elsewhere in this file
    monkeypatch.setattr(_bhfs, "_tol_to_niter", lambda tol: 1)
    with pytest.raises(ValueError, match=r"Newton residual reached.*N_R"):
        compute_bihalofit(
            _A0,
            _k1,
            _k1,
            _k1,
            pk_source="eh",
            tol=1e-8,
            validate=True,
            cosmo=fiducial_params,
        )

    # validate=False NaN-poisons exactly the offending entries instead.
    B = compute_bihalofit(
        _A0, _k1, _k1, _k1, pk_source="eh", tol=1e-8, cosmo=fiducial_params
    )
    assert np.all(np.isnan(np.asarray(B)))


def _force_bad_residual(monkeypatch):
    """Patch bispectrum_all_z so every call's k_NL/n_eff residual is
    forced to a large, never-converging value.

    An astronomically tiny tol (e.g. 1e-300) is *not* a portable way to
    force this path: once the Newton solve reaches its true fixed point,
    R_final can equal R_prev in an exact bit pattern, making the residual
    exactly 0.0 -- which trivially satisfies any tol regardless of how
    small. Whether that exact-fixed-point bit pattern is hit depends on the
    XLA compilation/autotune settings in effect, so this was observed to
    pass standalone but fail under pytest's conftest-configured XLA flags.
    Forcing the residual directly is deterministic and platform-independent.
    """
    orig_all_z = _bhfk.bispectrum_all_z

    def fake_all_z(*args, **kwargs):
        B, resid = orig_all_z(*args, **kwargs)
        return B, jnp.ones_like(resid)

    monkeypatch.setattr(_bhfk, "bispectrum_all_z", fake_all_z)


# Each fault-injection test below uses its own R_min, so that no compiled kernel is shared with
# another test and the monkeypatch always takes effect.
_R_MIN_BAD_NAN = 0.0501
_R_MIN_BAD_RAISE = 0.0502
_R_MIN_BAD_CACHING = 0.0503


def test_tol_unreachable_nans_by_default(monkeypatch, fiducial_params):
    """If the residual never drops below tol even at the N_R=30 cap,
    validate=False marks exactly those entries NaN rather than silently
    returning an under-converged value."""
    _force_bad_residual(monkeypatch)
    B = compute_bihalofit(
        _A0,
        _K1,
        _K1,
        _K1,
        pk_source="eh",
        tol=1e-8,
        R_min=_R_MIN_BAD_NAN,
        validate=False,
        cosmo=fiducial_params,
    )
    assert np.all(np.isnan(np.asarray(B)))


def test_tol_unreachable_raises_with_validate(monkeypatch, fiducial_params):
    """Same forced failure, but validate=True raises instead of NaN."""
    _force_bad_residual(monkeypatch)
    with pytest.raises(ValueError, match=r"Newton residual reached"):
        compute_bihalofit(
            _A0,
            _K1,
            _K1,
            _K1,
            pk_source="eh",
            tol=1e-8,
            R_min=_R_MIN_BAD_RAISE,
            validate=True,
            cosmo=fiducial_params,
        )


def test_validate_not_cached_across_different_values(monkeypatch, fiducial_params):
    """Building the model at the same geometry with validate=False, then
    validate=True, must not silently reuse a closure whose validate value
    was frozen in by the first call."""
    _force_bad_residual(monkeypatch)
    compute_bihalofit(
        _A0,
        _K1,
        _K1,
        _K1,
        pk_source="eh",
        tol=1e-8,
        R_min=_R_MIN_BAD_CACHING,
        validate=False,
        cosmo=fiducial_params,
    )
    with pytest.raises(ValueError, match=r"Newton residual reached"):
        compute_bihalofit(
            _A0,
            _K1,
            _K1,
            _K1,
            pk_source="eh",
            tol=1e-8,
            R_min=_R_MIN_BAD_CACHING,
            validate=True,
            cosmo=fiducial_params,
        )


def test_jacfwd_with_tol_and_validate(fiducial_params):
    """jax.jacfwd through compute_bihalofit with tol+validate set: the
    concrete primal call inside the jacfwd trace checks/retries as usual;
    the jacfwd-traced calls themselves skip verification (tracers have no
    concrete value to check), matching the documented jacfwd interaction."""
    p = dict(fiducial_params)

    def f(sigma8_cold):
        return compute_bihalofit(
            _A0,
            _K1,
            _K1,
            _K1,
            Cosmology(**{**p, "sigma8_cold": sigma8_cold}),
            pk_source="eh",
            tol=1e-8,
            validate=True,
        )

    B = f(p["sigma8_cold"])
    jac = jax.jacfwd(f)(p["sigma8_cold"])
    assert np.all(np.isfinite(np.asarray(B)))
    assert np.all(np.isfinite(np.asarray(jac)))


# ---------------------------------------------------------------------------
# term ('full'/'1h'/'3h'/'tree') and baryonic_boost
# ---------------------------------------------------------------------------

_SCALENE_K1 = np.array([0.30])
_SCALENE_K2 = np.array([0.20])
_SCALENE_K3 = np.array([0.48])


def test_term_invalid_raises(fiducial_params):
    """Unrecognised term raises ValueError before any JAX work."""
    with pytest.raises(ValueError, match="term must be one of"):
        compute_bihalofit(
            _A0, _K1, _K1, _K1, pk_source="eh", term="bogus", cosmo=fiducial_params
        )


def test_term_full_equals_1h_plus_3h(fiducial_params, test_triangles, redshifts):
    """term='full' is exactly the sum of term='1h' and term='3h', across a
    representative set of triangle shapes and redshifts -- not just a smoke
    test at one triangle, since a bug could plausibly only show up for
    specific shapes (e.g. a mis-ordered cyclic term)."""
    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)
    B_full = compute_bihalofit(
        a, k1, k2, k3, pk_source="eh", term="full", cosmo=fiducial_params
    )
    B_1h = compute_bihalofit(
        a, k1, k2, k3, pk_source="eh", term="1h", cosmo=fiducial_params
    )
    B_3h = compute_bihalofit(
        a, k1, k2, k3, pk_source="eh", term="3h", cosmo=fiducial_params
    )
    np.testing.assert_allclose(
        np.asarray(B_full), np.asarray(B_1h) + np.asarray(B_3h), rtol=1e-12
    )


def test_term_tree_matches_independent_pt_formula(fiducial_params):
    """term='tree' matches a hand-written 2nd-order PT bispectrum, using an
    independent (np.interp-based) log-log interpolation of the EH98 linear
    P(k) onto the standard k-grid -- independent of body()'s own F2/PL1-3
    implementation, catching e.g. a mis-ordered cyclic term that a
    self-consistency check against the library's own code could not.

    Note: P(k) must be evaluated by interpolating the full 600-point
    standard grid (matching what body() does internally), not by calling
    _pk_lin_eh_jax with a bespoke 3-point k_lin -- the sigma8 normalisation
    integral needs the full grid to be accurate; a 3-point grid silently
    mis-normalises P(k) by an O(1) factor.
    """
    from bihalofit_jax._pk_backends import _pk_lin_eh_jax, _STANDARD_K_GRID

    k1, k2, k3 = float(_SCALENE_K1[0]), float(_SCALENE_K2[0]), float(_SCALENE_K3[0])
    p = fiducial_params
    PL_std, _ = _pk_lin_eh_jax(
        jnp.asarray(p["omega_cold"]),
        jnp.asarray(p["omega_baryon"]),
        jnp.asarray(p["hubble"]),
        jnp.asarray(p["ns"]),
        jnp.asarray(p["sigma8_cold"]),
        jnp.asarray(p["w0"]),
        jnp.asarray(p["wa"]),
        np.array([1.0]),
        jnp.float64,
    )
    PL_std = np.asarray(PL_std)[0, 0]  # (N_k,)
    logk_grid = np.log10(_STANDARD_K_GRID)
    logPL_grid = np.log(PL_std)

    def P_at(k):
        return np.exp(np.interp(np.log10(k), logk_grid, logPL_grid))

    P1, P2, P3 = P_at(k1), P_at(k2), P_at(k3)

    def F2(ka, kb, kc):
        cos12 = (kc**2 - ka**2 - kb**2) / (2.0 * ka * kb)
        return 5.0 / 7.0 + 0.5 * cos12 * (ka / kb + kb / ka) + 2.0 / 7.0 * cos12**2

    B_ref = 2.0 * (
        F2(k1, k2, k3) * P1 * P2 + F2(k2, k3, k1) * P2 * P3 + F2(k3, k1, k2) * P3 * P1
    )

    B_tree = compute_bihalofit(
        _A0,
        _SCALENE_K1,
        _SCALENE_K2,
        _SCALENE_K3,
        pk_source="eh",
        term="tree",
        cosmo=fiducial_params,
    )
    np.testing.assert_allclose(float(np.asarray(B_tree).squeeze()), B_ref, rtol=1e-8)


def test_term_tree_not_poisoned_by_bracket_failure(fiducial_params):
    """term='tree' has no k_NL/n_eff dependence, so it stays finite even for
    a sigma8 so small that sigma^2(R) never reaches 1 (the same input that
    NaN-poisons 'full'/'1h'/'3h' -- see test_sigma2_guard)."""
    p = dict(fiducial_params)
    p["sigma8_cold"] = np.array([0.01])
    B_full = compute_bihalofit(_A0, _K1, _K1, _K1, cosmo=p, pk_source="eh", term="full")
    B_tree = compute_bihalofit(_A0, _K1, _K1, _K1, cosmo=p, pk_source="eh", term="tree")
    assert np.all(np.isnan(np.asarray(B_full)))
    assert np.all(np.isfinite(np.asarray(B_tree)))
    assert np.all(np.asarray(B_tree) > 0)


def test_term_tree_skips_knl_solve_and_never_retries_on_tol(
    monkeypatch, fiducial_params
):
    """term='tree' doesn't run the Newton solve at all (see body()'s
    dedicated early-return branch), so its reported residual is a constant
    zero -- this must never trigger make_bihalofit's retry-at-higher-N_R
    path, even with an absurdly tight tol and even if _tol_to_niter itself
    is broken (monkeypatched to under-estimate wildly, as in
    test_tol_validate_retry_recovers_full_precision)."""
    monkeypatch.setattr(_bhfs, "_tol_to_niter", lambda tol: 1)
    B = compute_bihalofit(
        _A0,
        _K1,
        _K1,
        _K1,
        pk_source="eh",
        tol=1e-300,
        validate=True,
        term="tree",
        cosmo=fiducial_params,
    )
    assert np.all(np.isfinite(np.asarray(B)))
    assert np.all(np.asarray(B) > 0)


def _reference_baryon_ratio(k1, k2, k3, z):
    """Independent re-implementation of Takahashi+19 App. C Eq. C1, ported
    from Gebauer_bihalofit's baryon_ratio (Numba reference) rather than
    copy-pasted from bihalofit_jax._baryon_ratio -- a real ground truth."""
    if z > 5.0:
        return 1.0
    a = 1.0 / (1.0 + z)
    A0 = 0.068 * (a - 0.5) ** 0.47 if a > 0.5 else 0.0
    mu0 = 0.018 * a + 0.837 * a * a
    sigma0 = 0.881 * mu0
    alpha0 = 2.346
    A1 = 1.052 * (a - 0.2) ** 1.41 if a > 0.2 else 0.0
    mu1 = abs(0.172 + 3.048 * a - 0.675 * a * a)
    sigma1 = (0.494 - 0.039 * a) * mu1
    ks = 29.90 - 38.73 * a + 24.30 * a * a
    alpha2 = 2.25
    beta2 = 0.563 / ((a / 0.060) ** 0.02 + 1.0) / alpha2
    Rb = 1.0
    for k in (k1, k2, k3):
        xi = np.log10(k)
        Rb *= (
            A0 * np.exp(-((abs(xi - mu0) / sigma0) ** alpha0))
            - A1 * np.exp(-((abs(xi - mu1) / sigma1) ** 2))
            + (1.0 + (k / ks) ** alpha2) ** beta2
        )
    return Rb


def test_baryonic_boost_matches_reference(fiducial_params):
    """baryonic_boost=True multiplies B by exactly the App. C ratio, checked
    against an independent re-implementation, at several redshifts."""
    for z in (0.0, 0.5, 1.0, 2.0):
        a = np.array([1.0 / (1.0 + z)])
        B_plain = compute_bihalofit(
            a,
            _SCALENE_K1,
            _SCALENE_K2,
            _SCALENE_K3,
            pk_source="eh",
            cosmo=fiducial_params,
        )
        B_boost = compute_bihalofit(
            a,
            _SCALENE_K1,
            _SCALENE_K2,
            _SCALENE_K3,
            pk_source="eh",
            baryonic_boost=True,
            cosmo=fiducial_params,
        )
        Rb_ref = _reference_baryon_ratio(
            float(_SCALENE_K1[0]), float(_SCALENE_K2[0]), float(_SCALENE_K3[0]), z
        )
        np.testing.assert_allclose(
            float(np.asarray(B_boost).squeeze()),
            float(np.asarray(B_plain).squeeze()) * Rb_ref,
            rtol=1e-10,
        )


def test_baryonic_boost_identity_beyond_z5(fiducial_params):
    """The App. C fit is uncalibrated beyond z=5, so baryonic_boost is a
    no-op there (Rb=1 exactly)."""
    a = np.array([1.0 / (1.0 + 6.0)])
    B_plain = compute_bihalofit(
        a, _SCALENE_K1, _SCALENE_K2, _SCALENE_K3, pk_source="eh", cosmo=fiducial_params
    )
    B_boost = compute_bihalofit(
        a,
        _SCALENE_K1,
        _SCALENE_K2,
        _SCALENE_K3,
        pk_source="eh",
        baryonic_boost=True,
        cosmo=fiducial_params,
    )
    np.testing.assert_array_equal(np.asarray(B_plain), np.asarray(B_boost))


def test_term_and_baryonic_boost_not_cached_across_different_values(fiducial_params):
    """Same triangle/redshift geometry built with different term/baryonic_boost
    combinations must not silently share a cached closure -- regression guard
    for the same cache-key bug class fixed earlier for tol/validate."""
    from bihalofit_jax import make_bihalofit

    f_full = make_bihalofit(_K1, _K1, _K1, _A0, pk_source="eh")
    f_1h = make_bihalofit(_K1, _K1, _K1, _A0, pk_source="eh", term="1h")
    f_tree = make_bihalofit(_K1, _K1, _K1, _A0, pk_source="eh", term="tree")
    f_full_boost = make_bihalofit(
        _K1, _K1, _K1, _A0, pk_source="eh", baryonic_boost=True
    )

    assert f_full is not f_1h
    assert f_full is not f_tree
    assert f_full is not f_full_boost

    p = {k: v for k, v in fiducial_params.items()}
    r_full = np.asarray(f_full(p))
    r_1h = np.asarray(f_1h(p))
    r_tree = np.asarray(f_tree(p))
    r_full_boost = np.asarray(f_full_boost(p))

    assert not np.allclose(r_full, r_1h)
    assert not np.allclose(r_full, r_tree)
    assert not np.allclose(r_full, r_full_boost)


def test_term_and_baryonic_boost_under_jacfwd(fiducial_params):
    """term and baryonic_boost compose cleanly under jax.jacfwd (differentiates
    through both the term selection and the boost multiplication)."""
    p = dict(fiducial_params)

    def f(sigma8_cold):
        return compute_bihalofit(
            _A0,
            _SCALENE_K1,
            _SCALENE_K2,
            _SCALENE_K3,
            Cosmology(**{**p, "sigma8_cold": sigma8_cold}),
            pk_source="eh",
            term="3h",
            baryonic_boost=True,
        )

    B = f(p["sigma8_cold"])
    jac = jax.jacfwd(f)(p["sigma8_cold"])
    assert np.all(np.isfinite(np.asarray(B)))
    assert np.all(np.isfinite(np.asarray(jac)))
    assert np.any(np.asarray(jac) != 0.0)


# ---------------------------------------------------------------------------
# k_lin (custom P(k) grid) -- controls the log-linear interpolation floor
# that off-grid k1/k2/k3 lookups (unlike the k_NL/n_eff solve, which only
# ever touches the grid's own nodes) cannot converge past. See
# docs -- the chi-quadrature convergence notebook that motivated this.
# ---------------------------------------------------------------------------


def test_k_lin_invalid_with_baccoemu_raises(fiducial_params):
    """k_lin is not supported with pk_source='baccoemu' (fixed emulator
    grid) -- rejected eagerly, before any JAX work."""
    k_lin = np.geomspace(1e-4, 50.0, 1200)
    with pytest.raises(ValueError, match="k_lin is not supported"):
        compute_bihalofit(
            _A0, _K1, _K1, _K1, pk_source="baccoemu", k_lin=k_lin, cosmo=fiducial_params
        )


def test_k_lin_default_none_matches_standard_grid(
    fiducial_params, test_triangles, redshifts
):
    """k_lin=None (default) must reproduce _STANDARD_K_GRID exactly -- the
    backward-compatibility guarantee everything else in the package relies
    on."""
    from bihalofit_jax._pk_backends import _STANDARD_K_GRID

    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)
    B_none = compute_bihalofit(
        a, k1, k2, k3, pk_source="eh", k_lin=None, cosmo=fiducial_params
    )
    B_explicit = compute_bihalofit(
        a, k1, k2, k3, pk_source="eh", k_lin=_STANDARD_K_GRID, cosmo=fiducial_params
    )
    np.testing.assert_array_equal(np.asarray(B_none), np.asarray(B_explicit))


def test_k_lin_finer_grid_shrinks_interpolation_error(
    fiducial_params, test_triangles, redshifts
):
    """A finer k_lin changes B (off-grid k1/k2/k3 interpolate against
    different neighbours) but stays close to the default-grid result --
    both should be converging toward the same true P_L(k), so the gap
    should be small, not merely finite."""
    k_lin_fine = np.geomspace(1e-4, 50.0, 4 * 600)
    k1, k2, k3 = test_triangles
    a = 1.0 / (1.0 + redshifts)
    B_default = np.asarray(
        compute_bihalofit(a, k1, k2, k3, pk_source="eh", cosmo=fiducial_params)
    )
    B_fine = np.asarray(
        compute_bihalofit(
            a, k1, k2, k3, pk_source="eh", k_lin=k_lin_fine, cosmo=fiducial_params
        )
    )
    assert np.all(np.isfinite(B_fine))
    rel = np.abs(B_fine / B_default - 1)
    assert np.nanmax(rel) < 1e-2  # generous -- this is a sanity bound, not a
    # precision claim; see the notebook for the
    # actual convergence-rate measurement


def test_k_lin_not_cached_across_different_grids(fiducial_params):
    """Same triangle/redshift geometry built with different k_lin must not
    silently share a cached closure -- same cache-key bug class already
    guarded against for tol/validate and term/baryonic_boost."""
    k_lin_fine = np.geomspace(1e-4, 50.0, 1200)
    f_default = make_bihalofit(_K1, _K1, _K1, _A0, pk_source="eh")
    f_fine = make_bihalofit(_K1, _K1, _K1, _A0, pk_source="eh", k_lin=k_lin_fine)
    assert f_default is not f_fine
    p = dict(fiducial_params)
    r_default = np.asarray(f_default(Cosmology(**p)))
    r_fine = np.asarray(f_fine(Cosmology(**p)))
    assert not np.allclose(r_default, r_fine, atol=0)


def test_k_lin_baccoemu_default_none_still_works(fiducial_params):
    """k_lin=None (the default) with pk_source='baccoemu' must NOT raise --
    only an explicit custom grid is rejected for that backend."""
    pytest.importorskip("baccoemu")
    B = compute_bihalofit(
        _A0, _K1, _K1, _K1, pk_source="baccoemu", k_lin=None, cosmo=fiducial_params
    )
    assert np.all(np.isfinite(np.asarray(B)))


# ---------------------------------------------------------------------------
# The default R_min/R_max bracket: the same in every builder, and wide enough for the whole
# documented prior box and redshift range (extreme low-h/high-z cosmologies need R_min well below 0.05).
# ---------------------------------------------------------------------------


def test_R_min_R_max_defaults_consistent_across_public_api():
    """The three builders all forward R_min/R_max into the same underlying
    _solve_knl_neff, so their defaults must agree -- otherwise a fix made to
    one (like widening R_min for extreme cosmologies) silently fails to
    protect the others. A plain signature-inspection check, not numerical."""
    import inspect
    from bihalofit_jax import make_bihalofit
    import bihalofit_jax.projected_bispectrum as _pbhf
    import bihalofit_jax.aperture as _apstat

    funcs = {
        "make_bihalofit": make_bihalofit,
        "make_projected_bihalofit": _pbhf.make_projected_bihalofit,
        "make_map3": _apstat.make_map3,
    }
    defaults = {}
    for name, fn in funcs.items():
        sig = inspect.signature(fn)
        assert "R_min" in sig.parameters, f"{name} lost its R_min parameter"
        assert "R_max" in sig.parameters, f"{name} lost its R_max parameter"
        defaults[name] = (
            sig.parameters["R_min"].default,
            sig.parameters["R_max"].default,
        )

    ref = defaults["make_bihalofit"]
    mismatched = {name: d for name, d in defaults.items() if d != ref}
    assert not mismatched, (
        f"R_min/R_max defaults disagree across the public API: "
        f"make_bihalofit has {ref}, but {mismatched} differ -- a caller "
        f"using one entry point instead of another would get different "
        f"(and possibly stale/unsafe) bracket-failure behaviour for the "
        f"same cosmology/redshift."
    )


def test_bihalofit_no_nan_at_known_extreme_case():
    """Deterministic regression test for the exact (cosmology, z) point that
    first exposed the R_min=0.05 bug: omega_cold=0.3592, sigma8_cold=0.6984,
    hubble=0.4705, ns=0.8541 at z=3.5 gives k_NL~30.8 h/Mpc (R_NL~0.032
    Mpc/h) in the reference C code, which fell just outside the old
    R_min=0.05 (k_NL<=20 h/Mpc) bracket. A randomized sweep might not
    reliably re-hit this exact corner by chance (it didn't, when this test
    suite was run against a deliberately-reverted R_min=0.05 during
    development) -- pin it explicitly so this specific regression can never
    silently reappear."""
    B = compute_bihalofit(
        np.array([1.0 / (1.0 + 3.5)]),
        np.array([0.1]),
        np.array([0.5]),
        np.array([1.0]),
        pk_source="eh",
        cosmo=Cosmology(
            omega_cold=np.array([0.3592]),
            omega_baryon=np.array([0.0492]),
            sigma8_cold=np.array([0.6984]),
            ns=np.array([0.8541]),
            hubble=np.array([0.4705]),
            neutrino_mass=np.array([0.0]),
            w0=np.array([-1.0]),
            wa=np.array([0.0]),
        ),
    )
    assert np.all(np.isfinite(np.asarray(B)))


def test_bihalofit_no_nan_across_prior_box_and_redshift():
    """Broad randomized sweep asserting compute_bihalofit at DEFAULT
    settings (R_min=1e-6, N_R=15) never returns NaN. This is exactly the
    scan that first found the R_min=0.05 bug (low h, high z) -- the
    fault-injection tests only exercise the bracket-failure mechanism when
    deliberately triggered, not whether the *default* bracket is wide
    enough for realistic inputs. Complements (does not replace)
    test_bihalofit_no_nan_at_known_extreme_case: a random sweep is not
    guaranteed to hit any single known-bad corner by chance, but adds
    broad coverage the one pinned case can't.

    Covers, deliberately wider than the first version of this test:
      - all 6 cosmological parameters that can plausibly push k_NL/n_eff
        toward a bracket edge (omega_cold, sigma8_cold, hubble, ns, w0, wa
        -- the same bounds as test_tol_to_niter_meets_claimed_accuracy's
        stress sweep, which already showed w0/wa matter for k_NL
        precision, not just h/sigma8), not just the 4 the original
        (accidentally narrower) version varied.
      - N=100 draws, not 15 -- each eval is ~1ms once compiled, so this is
        cheap; a larger N reduces (without eliminating -- see the pinned
        test) the chance a real regression is missed by chance the way the
        original N=15 version demonstrably was during development.
      - a mix of triangle shapes (equilateral, squeezed, scalene,
        near-degenerate), not just equilateral -- squeezed/near-degenerate
        triangles (r1/r2 -> 0) are exactly where the formula's alphan
        clipping and C1-smoothing (_smooth_min) live, a documented history
        of instability distinct from the k_NL/n_eff bracket issue this
        test otherwise targets.

    All cosmologies x all redshifts x all triangle shapes are batched into
    a single compute_bihalofit call (N_cosmo, N_z, N_tri all vectorised),
    not a Python-level loop -- both for speed and because it's a more
    faithful test of the batched-cosmology use case (e.g. an MCMC/HMC
    loop) this package is actually built for.
    """
    rng = np.random.default_rng(0)
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

    z_list = np.concatenate([[0.01], np.linspace(0.05, 3.0, 8)])
    a_list = 1.0 / (1.0 + z_list)

    # equilateral, squeezed, scalene, near-degenerate (k1 close to k2+k3)
    tri = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 0.1],
            [0.3, 0.5, 0.7],
            [1.0, 0.6, 0.45],
        ]
    )
    k1_, k2_, k3_ = tri[:, 0], tri[:, 1], tri[:, 2]

    B = compute_bihalofit(
        a_list,
        k1_,
        k2_,
        k3_,
        pk_source="eh",
        cosmo=Cosmology(
            omega_cold=draw["omega_cold"],
            omega_baryon=np.full(N, 0.0492),
            sigma8_cold=draw["sigma8_cold"],
            ns=draw["ns"],
            hubble=draw["hubble"],
            neutrino_mass=np.zeros(N),
            w0=draw["w0"],
            wa=draw["wa"],
        ),
    )
    B = np.asarray(B)  # (N_cosmo=100, N_z=9, N_tri=4)

    n_bad = int((~np.isfinite(B)).sum())
    assert n_bad == 0, (
        f"{n_bad}/{B.size} (cosmology, z, triangle) combination(s) out of "
        f"{N} cosmologies x {len(z_list)} redshifts x {len(tri)} triangle "
        "shapes returned non-finite output at default settings -- the "
        "default R_min/R_max bracket no longer covers the documented "
        "prior range."
    )
