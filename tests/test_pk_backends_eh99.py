"""
pytest suite for the 'eh_99' linear P(k) backend (Eisenstein & Hu 1999
CDM+HDM transfer function at f_nu=0, src/_pk_backends.py).

Run:
    pytest tests/test_pk_backends_eh99.py -v --device cpu
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bihalofit_jax._pk_backends import (
    _pk_backend,
    _VALID_SOURCES,
    _STANDARD_K_GRID,
    _W_SIGMA8,
    _eh99_mixed_dm_T,
    _eh_nowiggle_T,
)
from bihalofit_jax import Cosmology
from bihalofit_jax import smail_nz, make_projected_bihalofit

_OC, _OB, _S8, _NS, _H = 0.3156, 0.0492, 0.831, 0.9645, 0.6727
_LENSING3 = ("lensing", "lensing", "lensing")


def make_bkappa(ell1, ell2, ell3, n_z, **kwargs):
    """Backward-compatibility shim -- see test_projected_bihalofit.py's
    identical shim docstring."""
    return make_projected_bihalofit(
        ell1, ell2, ell3, (n_z, n_z, n_z), _LENSING3, **kwargs
    )


_A_SORTED = np.array([1.0, 0.6667, 0.5])  # descending scale factors


def _call_eh99(
    oc=_OC,
    ob=_OB,
    h=_H,
    ns=_NS,
    s8=_S8,
    w0=-1.0,
    wa=0.0,
    a_sorted=_A_SORTED,
    jdtype=jnp.float64,
    k_lin=None,
):
    pk = _pk_backend("eh_99", k_lin=k_lin)
    return pk(
        jnp.array([oc]),
        jnp.array([ob]),
        jnp.array([h]),
        jnp.array([ns]),
        jnp.array([s8]),
        jnp.array([0.0]),
        jnp.array([w0]),
        jnp.array([wa]),
        a_sorted,
        jdtype,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_eh_99_is_valid_source():
    assert "eh_99" in _VALID_SOURCES


def test_invalid_pk_source_message_lists_eh_99():
    """make_bkappa's ValueError for an unknown pk_source enumerates 'eh_99'."""
    z = np.linspace(0.0, 3.5, 50)
    nz = (z, smail_nz(z))
    with pytest.raises(ValueError, match="eh_99"):
        make_bkappa(
            np.array([100.0]),
            np.array([100.0]),
            np.array([100.0]),
            nz,
            pk_source="bogus",
        )


# ---------------------------------------------------------------------------
# Shape / finiteness
# ---------------------------------------------------------------------------


def test_eh_99_output_shape():
    PL_all, P_z0 = _call_eh99()
    N_z, N_k = len(_A_SORTED), len(_STANDARD_K_GRID)
    assert PL_all.shape == (N_z, 1, N_k)
    assert P_z0.shape == (1, N_k)


def test_eh_99_finite_and_positive():
    PL_all, P_z0 = _call_eh99()
    assert bool(jnp.all(jnp.isfinite(PL_all)))
    assert bool(jnp.all(PL_all > 0))
    assert bool(jnp.all(P_z0 > 0))


@pytest.mark.parametrize("N_cosmo", [1, 3])
def test_eh_99_batch_cosmo_shape(N_cosmo):
    pk = _pk_backend("eh_99")
    oc = jnp.full(N_cosmo, _OC)
    ob = jnp.full(N_cosmo, _OB)
    h = jnp.full(N_cosmo, _H)
    ns = jnp.full(N_cosmo, _NS)
    s8 = jnp.full(N_cosmo, _S8)
    mnu = jnp.zeros(N_cosmo)
    w0 = jnp.full(N_cosmo, -1.0)
    wa = jnp.zeros(N_cosmo)
    PL_all, P_z0 = pk(oc, ob, h, ns, s8, mnu, w0, wa, _A_SORTED, jnp.float64)
    assert PL_all.shape == (len(_A_SORTED), N_cosmo, len(_STANDARD_K_GRID))
    assert P_z0.shape == (N_cosmo, len(_STANDARD_K_GRID))


# ---------------------------------------------------------------------------
# sigma8 normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s8", [0.7, 0.831, 0.95])
def test_eh_99_sigma8_normalization(s8):
    """P(k, z=0) integrates (against the sigma8 window) back to sigma8_cold."""
    _, P_z0 = _call_eh99(s8=s8)
    sigma8_sq = jnp.einsum("k,k->", P_z0[0], jnp.array(_W_SIGMA8))
    assert float(jnp.sqrt(sigma8_sq)) == pytest.approx(s8, abs=1e-6)


# ---------------------------------------------------------------------------
# Growth factor reuse (shared machinery with 'eh')
# ---------------------------------------------------------------------------


def test_eh_99_growth_factor_matches_eh():
    """eh_99 reuses the same growth-factor machinery as 'eh' -- D(a)/D(1)
    should be identical between the two backends even though the transfer
    function (and hence P_z0 shape) differs."""
    PL_all_eh, P_z0_eh = _pk_backend("eh")(
        jnp.array([_OC]),
        jnp.array([_OB]),
        jnp.array([_H]),
        jnp.array([_NS]),
        jnp.array([_S8]),
        jnp.array([0.0]),
        jnp.array([-1.0]),
        jnp.array([0.0]),
        _A_SORTED,
        jnp.float64,
    )
    PL_all_99, P_z0_99 = _call_eh99()

    growth_eh = PL_all_eh[:, 0, :] / P_z0_eh[0, :][None, :]
    growth_99 = PL_all_99[:, 0, :] / P_z0_99[0, :][None, :]
    max_dev = float(jnp.max(jnp.abs(growth_eh - growth_99)))
    assert max_dev < 1e-10


# ---------------------------------------------------------------------------
# Distinctness from 'eh' (different fitting formula, despite shared name)
# ---------------------------------------------------------------------------


def test_eh_99_differs_from_eh():
    """eh_99 (EH99 CDM+HDM at f_nu=0) and eh (EH98 no-wiggle) use different
    fitting formulae and should not be numerically identical, but both are
    CDM+baryon-only transfer functions so the sigma8-normalized P(k) should
    stay within a modest tolerance of one another over the standard grid."""
    _, P_z0_eh = _pk_backend("eh")(
        jnp.array([_OC]),
        jnp.array([_OB]),
        jnp.array([_H]),
        jnp.array([_NS]),
        jnp.array([_S8]),
        jnp.array([0.0]),
        jnp.array([-1.0]),
        jnp.array([0.0]),
        _A_SORTED,
        jnp.float64,
    )
    _, P_z0_99 = _call_eh99()

    max_dev = float(jnp.max(jnp.abs(P_z0_99[0] / P_z0_eh[0] - 1.0)))
    assert 1e-6 < max_dev < 0.05, (
        f"eh_99 vs eh: max|ratio-1| = {max_dev:.4f} -- expected a small but "
        "non-zero difference (distinct fitting formulae, both CDM+baryon-only)"
    )


def test_eh99_transfer_function_distinct_from_eh_nowiggle():
    """The raw transfer functions themselves (before sigma8 renormalization)
    are not the same function of k."""
    k = jnp.array(_STANDARD_K_GRID)
    T_eh = _eh_nowiggle_T(k, jnp.array([_OC]), jnp.array([_OB]), jnp.array([_H]))
    T_99 = _eh99_mixed_dm_T(k, jnp.array([_OC]), jnp.array([_OB]), jnp.array([_H]))
    assert not bool(jnp.allclose(T_eh, T_99, rtol=1e-8))


# ---------------------------------------------------------------------------
# Custom k_lin support
# ---------------------------------------------------------------------------


def test_eh_99_custom_k_lin():
    k_custom = np.logspace(-3, 1, 50)
    PL_all, P_z0 = _call_eh99(k_lin=k_custom)
    assert PL_all.shape == (len(_A_SORTED), 1, 50)
    assert P_z0.shape == (1, 50)
    assert bool(jnp.all(jnp.isfinite(PL_all)))


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param", ["omega_cold", "omega_baryon", "hubble", "ns", "sigma8"]
)
def test_eh_99_gradient_finite(param):
    kwargs = dict(oc=_OC, ob=_OB, h=_H, ns=_NS, s8=_S8)
    key = {
        "omega_cold": "oc",
        "omega_baryon": "ob",
        "hubble": "h",
        "ns": "ns",
        "sigma8": "s8",
    }[param]

    def f(x):
        local = dict(kwargs)
        local[key] = x
        _, P_z0 = _call_eh99(**local)
        return jnp.sum(P_z0)

    g = jax.grad(f)(kwargs[key])
    assert np.isfinite(float(g))


# ---------------------------------------------------------------------------
# End-to-end through make_bkappa
# ---------------------------------------------------------------------------


def test_eh_99_through_make_bkappa():
    """eh_99 is usable end-to-end as a pk_source for the full bispectrum
    pipeline, not just in isolation."""
    z = np.linspace(0.0, 3.5, 200)
    nz = (z, smail_nz(z))
    ell = np.array([200.0, 500.0, 1000.0])
    bk = make_bkappa(ell, ell, ell, nz, pk_source="eh_99", N_chi=8)
    out = np.asarray(bk(Cosmology(_OC, _OB, _S8, _NS, _H)))
    assert out.shape == (1, 3)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)
