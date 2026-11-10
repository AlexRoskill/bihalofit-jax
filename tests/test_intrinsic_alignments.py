"""
Tests of the 'ia' (NLA intrinsic-alignment) tracer.

    pytest tests/test_intrinsic_alignments.py -v --device cpu

The integrated checks use the 'eh' backend (cheap; the tracer touches only the
kernels, never B_delta). Everything is float64 (pinned by conftest).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bihalofit_jax import Cosmology, IA, smail_nz, make_projected_bihalofit, make_map3
from bihalofit_jax._cosmology import as_ia
from bihalofit_jax.background import growth_ratios
from bihalofit_jax.tracers import (
    counts_kernel,
    ia_amplitude,
    ia_kernel,
)

_Z = np.linspace(0.0, 3.5, 300)
_DNDZ = (_Z, smail_nz(_Z))
_ELL = np.array([300.0])


def _cos(p):
    return Cosmology(**p)


def _build(tracers, **kw):
    return make_projected_bihalofit(
        _ELL, _ELL, _ELL, (_DNDZ,) * 3, tracers, pk_source="eh", N_chi=20, **kw
    )


# ---------------------------------------------------------------------------
# Growth factor and amplitude
# ---------------------------------------------------------------------------


def _lcdm_growth(a, Om):
    """D(a)/D(1) for flat LCDM from D ∝ H(a) ∫ da'/(a' H(a'))^3, by quadrature."""
    E = lambda x: np.sqrt(Om / x**3 + 1.0 - Om)

    def D(x):
        xs = np.linspace(1e-8, x, 400001)
        y = 1.0 / (xs * E(xs)) ** 3
        return E(x) * np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(xs))

    return D(a) / D(1.0)


def test_growth_ratios_match_closed_form_lcdm():
    Om = 0.3
    a = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    D = np.asarray(
        growth_ratios(jnp.asarray(a), jnp.asarray(Om), -1.0, 0.0, jnp.float64)
    )
    ref = np.array([_lcdm_growth(x, Om) for x in a])
    np.testing.assert_allclose(D, ref, rtol=1e-4)


def test_ia_amplitude_formula():
    z = np.array([0.1, 0.5, 1.0])
    D = np.array([0.95, 0.75, 0.6])
    A, eta, Om, zp = 1.3, -0.7, 0.3, 0.62
    f = np.asarray(ia_amplitude(Om, A, eta, zp, D, z))
    ref = -A * 0.0134 * Om / D * ((1 + z) / (1 + zp)) ** eta
    np.testing.assert_allclose(f, ref, rtol=1e-14)
    assert np.all(f < 0)  # A_IA > 0 anti-aligns


def test_ia_kernel_is_counts_kernel_with_amplitude_as_bias():
    """W_I = f_IA(z) n(z) H(z)/c: the counts kernel with the bias replaced by f_IA."""
    z_nodes = jnp.linspace(0.05, 2.0, 40)
    D = 1.0 / (1.0 + z_nodes) ** 0.76
    args = (jnp.asarray(_Z), jnp.asarray(_DNDZ[1]), len(_Z))
    W = ia_kernel(0.3, -1.0, 0.0, 1.0, 0.5, 0.62, D, z_nodes, *args)
    f = ia_amplitude(0.3, 1.0, 0.5, 0.62, D, z_nodes)
    ref = counts_kernel(0.3, -1.0, 0.0, f, z_nodes, *args)
    np.testing.assert_allclose(np.asarray(W), np.asarray(ref), rtol=1e-14)


# ---------------------------------------------------------------------------
# Projected bispectrum: scaling, sign, reduction to no-IA
# ---------------------------------------------------------------------------


def test_terms_scale_with_alignment_power_and_are_negative(fiducial_params):
    """With eta=0, GGI ∝ A, GII ∝ A^2, III ∝ A^3 exactly. The signs (negative for
    A>0 in GGI and III) guard against any positivity mask on the weights."""
    c = _cos(fiducial_params)
    ggi, gii, iii = (
        _build(t)
        for t in (
            ("lensing", "lensing", "ia"),
            ("lensing", "ia", "ia"),
            ("ia", "ia", "ia"),
        )
    )
    r = {
        A: [float(np.asarray(f(c, ia=IA(A)))[0, 0]) for f in (ggi, gii, iii)]
        for A in (1.0, 2.0, -1.0)
    }
    for k, power in enumerate((1, 2, 3)):
        np.testing.assert_allclose(r[2.0][k] / r[1.0][k], 2.0**power, rtol=1e-10)
        np.testing.assert_allclose(r[-1.0][k] / r[1.0][k], (-1.0) ** power, rtol=1e-10)
    assert r[1.0][0] < 0 and r[1.0][1] > 0 and r[1.0][2] < 0


def test_zero_amplitude_reproduces_no_ia(fiducial_params):
    c = _cos(fiducial_params)
    ggg = np.asarray(_build(("lensing",) * 3)(c))
    np.testing.assert_array_equal(
        np.asarray(_build(("lensing", "lensing", "ia"))(c, ia=IA(0.0))), 0.0
    )
    assert np.all(ggg > 0)


def test_leg_order_is_irrelevant(fiducial_params):
    c = _cos(fiducial_params)
    a = np.asarray(_build(("lensing", "lensing", "ia"))(c, ia=IA(1.0, 0.5)))
    b = np.asarray(_build(("ia", "lensing", "lensing"))(c, ia=IA(1.0, 0.5)))
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_eta_changes_signal_and_pivot_is_irrelevant_at_eta_zero(fiducial_params):
    c = _cos(fiducial_params)
    at = lambda z_pivot, eta: float(
        np.asarray(
            _build(("lensing", "lensing", "ia"), z_pivot=z_pivot)(c, ia=IA(1.0, eta))
        )[0, 0]
    )
    assert at(0.3, 0.0) == pytest.approx(at(0.62, 0.0), rel=1e-12)
    assert abs(at(0.62, 1.0) / at(0.62, 0.0) - 1.0) > 0.05
    assert abs(at(0.3, 1.0) / at(0.62, 1.0) - 1.0) > 0.05


def test_ia_batched_over_cosmologies(fiducial_params):
    p = {k: np.repeat(v, 3) for k, v in fiducial_params.items()}
    c = Cosmology(**p)
    f = _build(("lensing", "lensing", "ia"))
    A = np.array([0.5, 1.0, 2.0])
    B = np.asarray(f(c, ia=IA(A)))[:, 0]
    np.testing.assert_allclose(B / B[1], A / A[1], rtol=1e-10)


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


def test_derivatives_wrt_ia_and_cosmology_are_finite_and_match_fd(fiducial_params):
    c = _cos(fiducial_params)
    f = _build(("lensing", "ia", "ia"))
    fun = lambda A, eta: f(c, ia=IA(A, eta))[0, 0]
    dA, deta = jax.jacfwd(fun, argnums=(0, 1))(1.0, 0.5)
    h = 1e-5
    fdA = (fun(1.0 + h, 0.5) - fun(1.0 - h, 0.5)) / (2 * h)
    fdeta = (fun(1.0, 0.5 + h) - fun(1.0, 0.5 - h)) / (2 * h)
    np.testing.assert_allclose(float(dA), float(fdA), rtol=1e-6)
    np.testing.assert_allclose(float(deta), float(fdeta), rtol=1e-6)
    jac = jax.jacfwd(lambda cc: f(cc, ia=IA(1.0, 0.5)))(c)
    for name in Cosmology._fields:
        assert np.all(np.isfinite(np.asarray(getattr(jac, name)))), name


# ---------------------------------------------------------------------------
# Interface and validation
# ---------------------------------------------------------------------------


def test_ia_required_iff_an_ia_leg_exists(fiducial_params):
    c = _cos(fiducial_params)
    with pytest.raises(ValueError, match="ia is required"):
        _build(("lensing", "lensing", "ia"))(c)
    with pytest.raises(ValueError, match="ia must be None"):
        _build(("lensing",) * 3)(c, ia=IA(1.0))


def test_ia_tracer_rejects_bias_and_bad_shapes(fiducial_params):
    c = _cos(fiducial_params)
    with pytest.raises(ValueError):
        make_projected_bihalofit(
            _ELL,
            _ELL,
            _ELL,
            (_DNDZ,) * 3,
            ("lensing", "lensing", "ia"),
            bias=(None, None, 1.0),
            pk_source="eh",
        )
    with pytest.raises(ValueError, match="shape"):
        _build(("lensing", "lensing", "ia"))(c, ia=IA(np.ones(4)))
    with pytest.raises(ValueError, match="z_pivot"):
        _build(("lensing", "lensing", "ia"), z_pivot=-1.0)


def test_as_ia_accepts_common_forms_and_rejects_junk():
    assert as_ia(IA(1.0, 0.5)) == IA(1.0, 0.5)
    assert as_ia((1.0, 0.5)) == IA(1.0, 0.5)
    assert as_ia({"A_IA": 1.0}) == IA(1.0, 0.0)
    assert as_ia(1.0) == IA(1.0, 0.0)
    for bad in ("x", (1, 2, 3), {"foo": 1}, None):
        with pytest.raises(TypeError):
            as_ia(bad)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # coarse grid, chosen for speed
def test_map3_passes_ia_through(fiducial_params):
    c = _cos(fiducial_params)
    m = make_map3(
        (_DNDZ,) * 3,
        ("lensing", "lensing", "ia"),
        pk_source="eh",
        N_chi=20,
        N_ell=24,
        N_phi=16,
        ell_max=3000.0,
    )
    th = np.array([5.0]) * np.pi / 180 / 60
    v1 = np.asarray(m(th, th, th, c, ia=IA(1.0)))
    v2 = np.asarray(m(th, th, th, c, ia=IA(2.0)))
    np.testing.assert_allclose(v2, 2.0 * v1, rtol=1e-10)
    assert v1[0, 0] < 0
