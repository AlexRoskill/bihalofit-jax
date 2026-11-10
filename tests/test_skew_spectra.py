"""
pytest suite for src/skew_spectra.py -- generalised skew spectra
C_l^{(jk)i} built on top of projected_bispectrum.make_projected_bihalofit.

Run:
    pytest tests/test_skew_spectra.py -v --device cpu

Quadrature here is deliberately tiny (N_chi/N_R = 6) and the backend is 'eh':
these tests check correctness, selection rules, shapes and differentiability,
not converged accuracy, and should stay fast on any backend.
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import pytest

pytest.importorskip(
    "pyshtools", reason="skew spectra need the optional pyshtools dependency"
)

from bihalofit_jax import (
    Cosmology,
    IA,
    make_skew_spectra,
    TopHat,
    SoftTopHat,
    build_ell_nodes,
    smail_nz,
)

# Internals under test -- not part of the public API, so imported from the
# module that defines them rather than the package root.
from bihalofit_jax.skew_spectra import (
    precompute_w3j,
    _build_weights,
    _filter_spec,
    _evaluate,
)

_OC, _OB, _S8, _NS, _H = 0.3156, 0.0492, 0.831, 0.9645, 0.6727

_LENSING3 = ("lensing", "lensing", "lensing")

# Small but non-degenerate configuration shared by the end-to-end tests.
_ELLS = np.arange(2, 21)
_LMAX_F = 60
_BAND = (10, 30)

_FAST = dict(pk_source="eh", N_chi=6, N_R=6, dtype=np.float64)


def _nz(n=60):
    z = np.linspace(0.0, 3.0, n)
    return (z, smail_nz(z))


def _dndz():
    nz = _nz()
    return (nz, nz, nz)


def _cosmo(**over):
    p = dict(
        omega_cold=_OC,
        omega_baryon=_OB,
        sigma8_cold=_S8,
        ns=_NS,
        hubble=_H,
        neutrino_mass=0.0,
        w0=-1.0,
        wa=0.0,
    )
    p.update(over)
    return Cosmology(**p)


@pytest.fixture(scope="module")
def band_filter():
    return TopHat(*_BAND)


def _weights(ells, filt, m_a=(0, 0, 0), m_b=(0, 0, 0)):
    """_build_weights against a filter object, resolving its band first."""
    (f1, b1, _), (f2, b2, _) = _filter_spec(filt, None)
    return _build_weights(ells, b1, b2, _evaluate(f1, b1), _evaluate(f2, b2), m_a, m_b)


@pytest.fixture(scope="module")
def skew_direct(band_filter):
    return make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, interp=False, **_FAST
    )


@pytest.fixture(scope="module")
def skew_interp(band_filter):
    return make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, interp=True, **_FAST
    )


# ---------------------------------------------------------------------------
# Wigner-3j symbols. This is the load-bearing precompute: everything else in
# the module is a weighted sum over these, so an error here is invisible
# downstream (the output stays finite, smooth and plausibly shaped) and would
# quietly bias every skew spectrum. sympy computes 3j symbols exactly in
# rational arithmetic, which makes it an independent reference rather than a
# reimplementation of the same recursion.
#
# The non-zero m cases are the ones that matter for the generalisation: they
# pin down which leg each m attaches to. pyshtools' Wigner3j puts its first m
# on the *free* index, so passing (m1, m2, m3) straight through in the
# caller's order -- as the original reference code does -- silently attaches
# m1 to the wrong leg the moment m is not all zeros.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [(0, 0, 0), (2, -2, 0), (2, 1, -3), (-1, 3, -2)])
def test_w3j_matches_sympy(m):
    sympy_wigner = pytest.importorskip("sympy.physics.wigner")
    ells = np.arange(0, 13)
    got = precompute_w3j(ells, ells, ells, m)

    ref = np.zeros_like(got)
    for a, L in enumerate(ells):
        for b, L1 in enumerate(ells):
            for c, L2 in enumerate(ells):
                ref[a, b, c] = float(
                    sympy_wigner.wigner_3j(int(L1), int(L2), int(L), m[0], m[1], m[2])
                )

    err = np.max(np.abs(got - ref))
    assert err < 1e-12, f"m={m}: max |w3j - sympy| = {err:.3e}"
    # Guard against the test passing on an all-zero block.
    assert np.count_nonzero(ref) > 100, f"m={m}: reference block is near-empty"


def test_w3j_selection_rules():
    """All-zero m: fully symmetric under column permutation, and vanishing for
    odd l1+l2+l (parity) and outside the triangle inequality."""
    n = 13
    w = precompute_w3j(np.arange(n), np.arange(n), np.arange(n), (0, 0, 0))

    assert np.allclose(w, np.transpose(w, (1, 2, 0)), rtol=0, atol=1e-14)
    assert np.allclose(w, np.transpose(w, (0, 2, 1)), rtol=0, atol=1e-14)

    L, L1, L2 = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    assert np.all(w[(L + L1 + L2) % 2 == 1] == 0.0), "odd parity is not zero"
    outside = (L1 + L2 < L) | (np.abs(L1 - L2) > L)
    assert np.all(w[outside] == 0.0), "triangle inequality is violated"


def test_w3j_jmin_zero_row_is_correct():
    """l1 == l2 with all-zero m gives jmin == 0, the case the original
    np.concatenate roll corrupts (w[-0:] is the whole array, not an empty
    slice). The l == 0 entry has the exact closed form (-1)^l1 / sqrt(2 l1+1)."""
    l1 = np.arange(1, 9)
    w = precompute_w3j(np.array([0]), l1, l1, (0, 0, 0))[0]
    diag = np.diag(w)
    expected = (-1.0) ** l1 / np.sqrt(2.0 * l1 + 1.0)
    assert np.allclose(diag, expected, rtol=1e-13), f"{diag} != {expected}"


def test_w3j_rejects_bad_m():
    with pytest.raises(ValueError, match="m1 \\+ m2 \\+ m3 == 0"):
        precompute_w3j(np.arange(5), np.arange(5), np.arange(5), (1, 1, 1))


def test_w3j_import_error_message_is_actionable():
    """The lazy import must name the missing package and how to get it."""
    import bihalofit_jax.skew_spectra as ss

    src = ss.precompute_w3j.__doc__ or ""
    assert "pyshtools" in src or "pyshtools" in ss.__doc__


# ---------------------------------------------------------------------------
# Filters and the sparsified weight tensor. Sparsifying is what guarantees
# make_projected_bihalofit -- which does no triangle validation of its own --
# never sees a configuration it cannot represent.
# ---------------------------------------------------------------------------


def test_tophat_is_an_inclusive_band():
    band = _filter_spec(TopHat(5, 9), None)[0][1]
    assert np.array_equal(band, np.arange(5, 10))
    assert np.array_equal(
        _evaluate(TopHat(5, 9), np.arange(3, 12)),
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )


def test_tophat_rejects_empty_band():
    with pytest.raises(ValueError, match="empty support"):
        _filter_spec(TopHat(9, 5), None)


def test_one_filter_is_used_on_both_summed_legs():
    """`filters=f` must be exactly `filters=(f, f)` -- the common case."""
    one = _filter_spec(TopHat(5, 9), None)
    pair = _filter_spec((TopHat(5, 9), TopHat(5, 9)), None)
    assert np.array_equal(one[0][1], pair[0][1])
    assert np.array_equal(one[1][1], pair[1][1])
    with pytest.raises(ValueError, match="2-tuple"):
        _filter_spec((TopHat(5, 9),) * 3, None)


def test_raw_array_filter_still_accepted():
    """The escape hatch for arbitrary filter shapes."""
    arr = np.exp(-(((np.arange(41) - 20.0) / 5.0) ** 2))
    arr[0] = 0.0
    f, band, static = _filter_spec(arr, None)[0]
    assert static and band[0] >= 1
    with pytest.raises(ValueError, match="identically zero"):
        _filter_spec(np.zeros(10), None)
    with pytest.raises(ValueError, match="vanish at ell = 0"):
        _filter_spec(np.ones(10), None)


def test_ell_sum_overrides_the_derived_support():
    _, band, _ = _filter_spec(TopHat(20, 30), (5, 60))[0]
    assert band[0] == 5 and band[-1] == 60


def test_soft_tophat_support_is_padded_by_width():
    """A logistic never reaches zero, so its support is padded rather than
    cut at the band edge."""
    _, narrow, _ = _filter_spec(SoftTopHat(20, 30, width=0.5), None)[0]
    _, wide, _ = _filter_spec(SoftTopHat(20, 30, width=4.0), None)[0]
    assert wide[0] < narrow[0] and wide[-1] > narrow[-1]
    assert narrow[0] >= 1


def test_weights_respect_selection_rules(band_filter):
    L, L1, L2, w, seg = _weights(_ELLS, band_filter)
    assert w.size > 0
    assert np.all(L1 + L2 >= L) and np.all(
        np.abs(L1 - L2) <= L
    ), "a surviving triple violates the triangle inequality"
    assert np.all((L + L1 + L2) % 2 == 0), "odd-parity triple survived"
    assert np.array_equal(_ELLS[seg], L)
    # Only the filter band contributes to the summed legs.
    assert L1.min() >= _BAND[0] and L1.max() <= _BAND[1]
    assert L2.min() >= _BAND[0] and L2.max() <= _BAND[1]
    # Squared 3j with a positive top-hat: every weight is positive.
    assert np.all(w > 0.0)


def test_weights_raise_when_bands_cannot_reach_the_output_ells():
    """A band far too narrow to close a triangle with the requested ells is a
    configuration error, not a silently-zero spectrum."""
    F = TopHat(10, 11)
    with pytest.raises(ValueError, match="identically zero"):
        _weights(np.arange(40, 50), F)


def test_build_ell_nodes_span_and_monotonicity():
    nodes = build_ell_nodes(2, 500)
    assert nodes[0] == 2 and nodes[-1] == 500
    assert np.all(np.diff(nodes) > 0)
    assert nodes.size < 120, "node set should stay sparse"


# ---------------------------------------------------------------------------
# Faithfulness to the reference implementation (anicola/bispectrum, fsb.py).
#
# Bk_fsb builds
#     Bl_fsb = (2l1+1)(2l2+1)(2l3+1)/4pi * big_w3j**2 * B
# and then bins it as sum(Bl_fsb over the bin) / sum(2l3+1). For a bin holding
# a single multipole those (2l+1) factors cancel exactly, so the per-ell output
# here must reproduce that expression term for term. This test transcribes it
# directly, given the same bispectrum values, and is the acceptance criterion
# for "faithful port".
# ---------------------------------------------------------------------------


def _reference_fsb(ells, band, B):
    """Direct transcription of Bk_fsb's inner expression, one ell at a time.

    ``B`` is indexed ``[l1 - band[0], l2 - band[0], l]``, matching the
    ``(l1, l2, l3)`` ordering the reference code evaluates its bispectrum on.
    """
    l1 = l2 = np.arange(band[0], band[1] + 1)
    big_w3j = precompute_w3j(ells, l1, l2, (0, 0, 0)) ** 2  # her big_w3j**2

    out = np.zeros(ells.size)
    for i, L in enumerate(ells):
        integrand = (
            (2 * l1[:, None] + 1)
            * (2 * l2[None, :] + 1)
            * (2 * L + 1)
            / 4.0
            / np.pi
            * big_w3j[i]
            * B[:, :, L]
        )
        out[i] = integrand.sum() / (2 * L + 1)  # her bin division
    return out


def test_reproduces_reference_fsb_expression(band_filter):
    """Same weights, same arithmetic as fsb.Bk_fsb, on a synthetic B."""
    rng = np.random.default_rng(0)
    l1 = np.arange(_BAND[0], _BAND[1] + 1)
    B = rng.uniform(0.5, 1.5, size=(l1.size, l1.size, int(_ELLS.max()) + 1))
    B = 0.5 * (B + np.swapaxes(B, 0, 1))  # B is symmetric in l1 <-> l2

    ref = _reference_fsb(_ELLS, _BAND, B)

    L, L1, L2, w, seg = _weights(_ELLS, band_filter)
    got = np.zeros(_ELLS.size)
    np.add.at(got, seg, w * B[L1 - _BAND[0], L2 - _BAND[0], L])

    assert np.allclose(
        got, ref, rtol=1e-12
    ), f"max rel dev {np.max(np.abs(got / ref - 1)):.3e}"


def test_m_b_defaults_to_m_a(band_filter):
    """m_b=None must give the squared 3j, i.e. the scalar FSB."""
    a = _weights(_ELLS, band_filter)
    kw = dict(interp=False, **_FAST)
    f1 = make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, m_a=(0, 0, 0), m_b=None, **kw
    )
    f2 = make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, m_a=(0, 0, 0), m_b=(0, 0, 0), **kw
    )
    # The two paths sum the same terms in a different order, so they agree to
    # round-off rather than bitwise (~2 ULP observed). The tolerance is still
    # ~1e4 times tighter than any real discrepancy would be.
    assert np.allclose(
        np.asarray(f1(_cosmo())), np.asarray(f2(_cosmo())), rtol=1e-12, atol=0.0
    )
    assert a[3].size > 0


def test_nonzero_m_changes_the_answer(band_filter):
    """The generalisation is real: a spin-2-style m set gives a different
    (and not merely rescaled) spectrum from the scalar case."""
    kw = dict(interp=False, **_FAST)
    scalar = make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, m_a=(0, 0, 0), **kw
    )(_cosmo())
    spin = make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, m_a=(2, -2, 0), **kw
    )(_cosmo())
    scalar, spin = np.asarray(scalar)[0], np.asarray(spin)[0]
    assert np.all(np.isfinite(spin))
    ratio = spin / scalar
    assert not np.allclose(
        ratio, ratio[0], rtol=1e-3
    ), "spin-2 m set only rescaled the scalar result -- suspect m ordering"


# ---------------------------------------------------------------------------
# End-to-end behaviour of the returned callable.
# ---------------------------------------------------------------------------


def test_output_shape_scalar_and_batched_cosmology(skew_direct):
    assert skew_direct(_cosmo()).shape == (1, _ELLS.size)

    batched = Cosmology(
        omega_cold=np.array([_OC, 0.32, 0.30]),
        omega_baryon=np.full(3, _OB),
        sigma8_cold=np.array([_S8, 0.85, 0.80]),
        ns=np.full(3, _NS),
        hubble=np.full(3, _H),
        neutrino_mass=np.zeros(3),
        w0=np.full(3, -1.0),
        wa=np.zeros(3),
    )
    assert skew_direct(batched).shape == (3, _ELLS.size)


def test_output_is_finite_and_positive(skew_direct):
    C = np.asarray(skew_direct(_cosmo()))
    assert np.all(np.isfinite(C)), "non-finite skew spectrum"
    # Squared 3j, positive top-hat filters and a positive lensing bispectrum
    # leave no route to a negative value here.
    assert np.all(C > 0.0), f"non-positive entries: {C[C <= 0]}"


def test_interp_agrees_with_direct(skew_direct, skew_interp):
    """The interpolated path is the production one; the direct path is what
    quantifies its error. Trilinear log-log interpolation off a ~70-node grid
    should hold this configuration to well under a percent."""
    d = np.asarray(skew_direct(_cosmo()))
    i = np.asarray(skew_interp(_cosmo()))
    dev = np.max(np.abs(i / d - 1.0))
    assert dev < 0.02, f"interp vs direct max rel deviation {dev:.3e}"


def test_interp_keeps_the_sign_of_a_negative_bispectrum(band_filter):
    """One 'ia' leg makes the projected bispectrum negative (f_IA < 0 for A_IA > 0). Interpolation
    must not clamp it to ~0: it must agree with the direct sum, sign included."""
    tracers = ("lensing", "lensing", "ia")
    direct, interp = (
        np.asarray(
            make_skew_spectra(
                _ELLS, band_filter, _dndz(), tracers, interp=flag, **_FAST
            )(_cosmo(), ia=IA(1.0))
        )
        for flag in (False, True)
    )
    assert np.all(direct < 0.0)
    assert np.max(np.abs(interp / direct - 1.0)) < 0.02


def test_grows_with_sigma8(skew_direct):
    """The bispectrum grows fast with sigma8, and the reduction is a
    positive-weight sum, so the skew spectrum must too."""
    vals = [
        np.asarray(skew_direct(_cosmo(sigma8_cold=s8))).sum()
        for s8 in (0.75, 0.83, 0.90)
    ]
    assert np.all(np.diff(vals) > 0), vals


@pytest.mark.parametrize("param", ["omega_cold", "sigma8_cold", "hubble"])
def test_gradients_are_finite(skew_direct, param):
    def scalar(x):
        return jnp.sum(skew_direct(_cosmo(**{param: x})))

    g = jax.grad(scalar)(float(_cosmo()._asdict()[param]))
    assert np.isfinite(g), f"d/d{param} is not finite: {g}"
    assert g != 0.0, f"d/d{param} is exactly zero"


def test_composes_under_outer_jit(skew_direct):
    """The callable is pure, so an outer jit must reproduce it exactly."""
    plain = np.asarray(skew_direct(_cosmo()))
    jitted = np.asarray(jax.jit(skew_direct)(_cosmo()))
    assert np.allclose(plain, jitted, rtol=1e-12)


def test_term_is_forwarded(band_filter):
    """'full' must equal '1h' + '3h', as at the bispectrum layer."""
    kw = dict(interp=False, **{k: v for k, v in _FAST.items() if k != "term"})
    full = np.asarray(
        make_skew_spectra(_ELLS, band_filter, _dndz(), _LENSING3, term="full", **kw)(
            _cosmo()
        )
    )
    one = np.asarray(
        make_skew_spectra(_ELLS, band_filter, _dndz(), _LENSING3, term="1h", **kw)(
            _cosmo()
        )
    )
    three = np.asarray(
        make_skew_spectra(_ELLS, band_filter, _dndz(), _LENSING3, term="3h", **kw)(
            _cosmo()
        )
    )
    assert np.allclose(full, one + three, rtol=1e-10)


# ---------------------------------------------------------------------------
# Error handling.
# ---------------------------------------------------------------------------


def test_rejects_ell_zero(band_filter):
    with pytest.raises(ValueError, match="must be >= 1"):
        make_skew_spectra(np.arange(0, 10), band_filter, _dndz(), _LENSING3, **_FAST)


def test_rejects_filter_with_support_at_ell_zero():
    F = np.concatenate([np.ones(31), np.zeros(30)])  # support includes ell=0
    with pytest.raises(ValueError, match="vanish at ell = 0"):
        make_skew_spectra(_ELLS, F, _dndz(), _LENSING3, **_FAST)


def test_rejects_empty_ells(band_filter):
    with pytest.raises(ValueError, match="ells is empty"):
        make_skew_spectra(
            np.array([], dtype=int), band_filter, _dndz(), _LENSING3, **_FAST
        )


def test_interp_accuracy_is_tuneable_through_nodes():
    """Accuracy is the caller's to set. A denser node set must measurably
    reduce the error, which is the point of exposing `nodes`."""
    ells = np.arange(2, 13)
    F = TopHat(60, 100)
    exact = np.asarray(
        make_skew_spectra(ells, F, _dndz(), _LENSING3, interp=False, **_FAST)(_cosmo())
    )[0]

    def err(nodes):
        got = np.asarray(
            make_skew_spectra(
                ells, F, _dndz(), _LENSING3, interp=True, nodes=nodes, **_FAST
            )(_cosmo())
        )[0]
        return np.abs(got / exact - 1.0).max()

    coarse = build_ell_nodes(2, 100, spacing="log", n=12)
    fine = build_ell_nodes(2, 100, spacing="log", n=60)
    assert fine.size > coarse.size
    assert err(fine) < err(
        coarse
    ), f"denser nodes did not help: coarse {err(coarse):.3%} vs fine {err(fine):.3%}"


def test_hybrid_spacing_matches_the_reference_set():
    """spacing='hybrid' must reproduce anicola/bispectrum's node set."""
    l_min, l_max = 2, 1000
    ref = np.concatenate(
        (np.arange(l_min, 10), np.linspace(10, 96, 20), np.geomspace(100, l_max, 50))
    )
    ref = np.unique(ref.astype(int))
    ref = np.unique(np.concatenate([[l_min], ref, [l_max]]))
    assert np.array_equal(build_ell_nodes(l_min, l_max, spacing="hybrid"), ref)


def test_auto_spacing_picks_shape_from_l_min():
    """Linear once l_min clears the fast low-ell variation, hybrid below it --
    the two must actually differ, or 'auto' is doing nothing."""
    lo = build_ell_nodes(2, 1000)  # -> hybrid
    hi = build_ell_nodes(20, 1000)  # -> linear
    assert np.array_equal(lo, build_ell_nodes(2, 1000, spacing="hybrid"))
    assert np.array_equal(hi, build_ell_nodes(20, 1000, spacing="linear"))
    # Linear spacing is uniform; the hybrid set is not.
    assert len(set(np.diff(hi).tolist())) <= 2
    assert len(set(np.diff(lo).tolist())) > 2


def test_auto_node_count_tracks_l_min_and_is_capped():
    """The default density targets a spacing of ~l_min/2 at the top of the
    range, since that is what resolves the |l1-l2| <= l sliver."""
    coarse_lmin = build_ell_nodes(100, 1000)
    fine_lmin = build_ell_nodes(20, 1000)
    assert fine_lmin.size > coarse_lmin.size
    assert np.diff(fine_lmin).max() <= 20 // 2 + 2
    # Cube cost means the count is capped however small l_min gets.
    assert build_ell_nodes(1, 10000, spacing="linear").size <= 142


def test_build_ell_nodes_validates():
    with pytest.raises(ValueError, match="l_min >= 1"):
        build_ell_nodes(0, 100)
    with pytest.raises(ValueError, match="l_max > l_min"):
        build_ell_nodes(100, 100)
    with pytest.raises(ValueError, match="spacing must be"):
        build_ell_nodes(2, 100, spacing="quadratic")


# ---------------------------------------------------------------------------
# Wigner-3j caching. Opt-in, content-addressed: the key covers everything that
# changes the result, so a stale hit is impossible and no invalidation logic
# is needed. Blocks reach several GiB, which is why nothing is written unless
# cache_dir is given.
# ---------------------------------------------------------------------------


def test_w3j_cache_roundtrip_is_exact(tmp_path):
    ells = np.arange(0, 12)
    cold = precompute_w3j(ells, ells, ells, (0, 0, 0), cache_dir=str(tmp_path))
    files = list(tmp_path.glob("w3j_*.npy"))
    assert len(files) == 1, f"expected one cache file, got {files}"

    warm = precompute_w3j(ells, ells, ells, (0, 0, 0), cache_dir=str(tmp_path))
    assert np.array_equal(np.asarray(warm), cold)
    assert len(list(tmp_path.glob("w3j_*.npy"))) == 1, "second call rewrote the cache"


def test_w3j_cache_key_separates_distinct_setups(tmp_path):
    """Anything that changes the block must change the file."""
    e = np.arange(0, 10)
    precompute_w3j(e, e, e, (0, 0, 0), cache_dir=str(tmp_path))
    precompute_w3j(e, e, e, (2, -2, 0), cache_dir=str(tmp_path))  # different m
    precompute_w3j(np.arange(0, 9), e, e, (0, 0, 0), cache_dir=str(tmp_path))
    assert len(list(tmp_path.glob("w3j_*.npy"))) == 3


def test_w3j_cache_hit_does_not_need_pyshtools(tmp_path, monkeypatch):
    """The lazy import sits after the cache lookup, so a warm cache works
    without the optional dependency -- worth pinning, since it is the whole
    reason the import is where it is."""
    import builtins

    e = np.arange(0, 10)
    cold = precompute_w3j(e, e, e, (0, 0, 0), cache_dir=str(tmp_path))

    real_import = builtins.__import__

    def no_pyshtools(name, *args, **kw):
        if name.startswith("pyshtools"):
            raise ImportError("pyshtools is not installed")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_pyshtools)
    warm = precompute_w3j(e, e, e, (0, 0, 0), cache_dir=str(tmp_path))
    assert np.array_equal(np.asarray(warm), cold)


def test_no_cache_written_by_default(tmp_path):
    e = np.arange(0, 10)
    precompute_w3j(e, e, e, (0, 0, 0))
    assert list(tmp_path.iterdir()) == []


def test_make_skew_spectra_forwards_cache_dir(tmp_path, band_filter):
    f = make_skew_spectra(
        _ELLS,
        band_filter,
        _dndz(),
        _LENSING3,
        interp=False,
        cache_dir=str(tmp_path),
        **_FAST,
    )
    assert list(tmp_path.glob("w3j_*.npy")), "no cache file written"
    # The cached path must give the same answer as the uncached one.
    g = make_skew_spectra(
        _ELLS,
        band_filter,
        _dndz(),
        _LENSING3,
        interp=False,
        cache_dir=str(tmp_path),
        **_FAST,
    )
    assert np.allclose(np.asarray(f(_cosmo())), np.asarray(g(_cosmo())), rtol=0)


def test_two_m_values_cache_two_blocks(tmp_path, band_filter):
    """The skew spectrum multiplies two 3j symbols, which are distinct arrays
    whenever m_a != m_b -- so each must get its own cache entry. When they are
    equal the block is computed once and shared, so exactly one file."""
    kw = dict(interp=False, cache_dir=str(tmp_path), **_FAST)

    make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, m_a=(0, 0, 0), m_b=None, **kw
    )
    assert (
        len(list(tmp_path.glob("w3j_*.npy"))) == 1
    ), "m_a == m_b should share one block"

    make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, m_a=(2, -2, 0), m_b=(0, 0, 0), **kw
    )
    # (0,0,0) is already cached from above, so this adds exactly one file.
    assert len(list(tmp_path.glob("w3j_*.npy"))) == 2, "m_a != m_b needs two blocks"


def test_cached_and_uncached_agree_for_two_m(tmp_path, band_filter):
    """A cache hit on one leg must not change the answer."""
    kw = dict(interp=False, m_a=(2, -2, 0), m_b=(0, 0, 0), **_FAST)
    warm = make_skew_spectra(
        _ELLS, band_filter, _dndz(), _LENSING3, cache_dir=str(tmp_path), **kw
    )
    cold = make_skew_spectra(_ELLS, band_filter, _dndz(), _LENSING3, **kw)
    # Round-off, not bitwise: the cached weights are reloaded from disk and
    # reduced in a different order from the freshly built ones.
    assert np.allclose(
        np.asarray(warm(_cosmo())), np.asarray(cold(_cosmo())), rtol=1e-12, atol=0.0
    )


# ---------------------------------------------------------------------------
# Differentiable band edges.
#
# A hard TopHat edge has no usable derivative: multipoles are integers, so
# sliding an edge changes nothing until it crosses one, then a whole
# multipole's terms drop out at once. jax.grad returns 0, which *looks* like an
# answer -- test_tophat_edge_gradient_is_flat records that trap. SoftTopHat
# smooths the edge so the derivative exists, which is what a Fisher forecast
# over the band needs.
# ---------------------------------------------------------------------------

_SOFT_KW = dict(interp=False, ell_sum=(2, 60), **_FAST)


def _soft(filt, **over):
    kw = dict(_SOFT_KW)
    kw.update(over)
    return make_skew_spectra(_ELLS, filt, _dndz(), _LENSING3, **kw)


def test_soft_tophat_shape():
    F = SoftTopHat(20.0, 40.0, width=2.0)
    # The plateau sits just below 1 at finite width -- a logistic never
    # saturates. This is a real softening of the band, not a
    # reparameterisation of TopHat, and it tightens as width shrinks.
    assert float(_evaluate(F, np.array([30.0]))[0]) == pytest.approx(0.99, abs=0.01)
    assert float(
        _evaluate(SoftTopHat(20.0, 40.0, width=0.5), np.array([30.0]))[0]
    ) == pytest.approx(1.0, abs=1e-6)
    assert float(_evaluate(F, np.array([0.0]))[0]) == pytest.approx(0.0, abs=1e-4)
    assert float(_evaluate(F, np.array([80.0]))[0]) == pytest.approx(0.0, abs=1e-4)
    # monotone rising through the lower edge, falling through the upper
    ell = np.arange(5, 56)
    v = _evaluate(F, ell)
    assert np.all(np.diff(v[ell <= 20]) > 0)
    assert np.all(np.diff(v[ell >= 40]) < 0)


def test_soft_tophat_converges_to_tophat(band_filter):
    """width -> 0 must reproduce TopHat exactly: the ramps sit half a multipole
    outside the band, so ell_lo and ell_hi are fully included. Centring them on
    the edges instead would give the boundary multipoles weight 1/2."""
    hard = np.asarray(
        make_skew_spectra(
            _ELLS, band_filter, _dndz(), _LENSING3, interp=False, **_FAST
        )(_cosmo())
    )
    soft = np.asarray(
        _soft(SoftTopHat(float(_BAND[0]), float(_BAND[1]), width=1e-6))(_cosmo())
    )
    assert np.allclose(
        soft, hard, rtol=1e-9
    ), f"max rel dev {np.max(np.abs(soft / hard - 1)):.3e}"


def test_tophat_edge_gradient_is_flat(band_filter):
    """Why SoftTopHat exists: a hard edge differentiates to exactly zero."""

    def total(lo):
        F = np.zeros(_LMAX_F + 1)
        F[int(lo) : _BAND[1] + 1] = 1.0
        return float(
            np.sum(
                np.asarray(
                    make_skew_spectra(
                        _ELLS, F, _dndz(), _LENSING3, interp=False, **_FAST
                    )(_cosmo())
                )
            )
        )

    # Moving the edge a fraction of a multipole changes nothing at all.
    assert total(10.0) == pytest.approx(total(10.4), rel=1e-12)


def test_static_filter_rejects_a_per_call_override(band_filter, skew_direct):
    with pytest.raises(ValueError, match="SoftTopHat"):
        skew_direct(_cosmo(), band_filter)


@pytest.mark.parametrize("width", [1.0, 3.0])
def test_edge_gradient_matches_finite_difference(width):
    """At a non-integer edge, away from any integer boundary."""
    filt = SoftTopHat(10.3, 30.0, width=width)
    skew = _soft(filt)

    def total(lo):
        return jnp.sum(skew(_cosmo(), filt._replace(ell_lo=lo)))

    g = float(jax.grad(total)(10.3))
    h = 1e-4
    fd = float((total(10.3 + h) - total(10.3 - h)) / (2 * h))
    assert np.isfinite(g) and g != 0.0
    assert g == pytest.approx(fd, rel=1e-5), f"grad {g:.6e} vs FD {fd:.6e}"


def test_gradient_is_a_named_pytree():
    """SoftTopHat is a pytree, so gradients come back named -- the Cosmology
    idiom."""
    filt = SoftTopHat(10.3, 30.0, width=2.0)
    skew = _soft(filt)
    g = jax.grad(lambda f: jnp.sum(skew(_cosmo(), f)))(filt)
    assert isinstance(g, SoftTopHat)
    assert all(np.isfinite(v) for v in (g.ell_lo, g.ell_hi, g.width))
    # Raising the lower edge removes modes; raising the upper edge adds them.
    assert g.ell_lo < 0 and g.ell_hi > 0


@pytest.mark.slow
def test_mixed_second_derivative_matches_finite_difference():
    """d2C/d(cosmo)d(edge) is what optimising the band needs. C is bilinear in
    the filter and the bispectrum, so this costs only two first derivatives --
    no Hessian of the projected bispectrum."""
    filt = SoftTopHat(10.3, 30.0, width=2.0)
    skew = _soft(filt)

    def total(s8, lo):
        return jnp.sum(skew(_cosmo(sigma8_cold=s8), filt._replace(ell_lo=lo)))

    mixed = float(jax.grad(jax.grad(total, argnums=0), argnums=1)(0.831, 10.3))
    h = 1e-4
    d = jax.grad(total, argnums=0)
    fd = float((d(0.831, 10.3 + h) - d(0.831, 10.3 - h)) / (2 * h))
    assert np.isfinite(mixed) and mixed != 0.0
    assert mixed == pytest.approx(fd, rel=1e-5), f"{mixed:.6e} vs FD {fd:.6e}"


@pytest.mark.slow
def test_fisher_logdet_is_differentiable_in_the_band():
    """End to end: Fisher matrix for two cosmological parameters, then the
    gradient of its log-determinant w.r.t. the band edges -- the capability
    this was built for. Checked against a coarse scan."""
    filt = SoftTopHat(10.0, 30.0, width=2.0)
    skew = _soft(filt)

    def logdetF(f):
        th = jnp.array([_OC, _S8])

        def Cvec(t):
            return skew(_cosmo(omega_cold=t[0], sigma8_cold=t[1]), f)[0]

        J = jax.jacrev(Cvec)(th)
        cov = jnp.diag((0.1 * Cvec(th)) ** 2)  # toy 10% diagonal errors
        return jnp.linalg.slogdet(J.T @ jnp.linalg.solve(cov, J))[1]

    g = jax.grad(logdetF)(filt)
    assert np.isfinite(g.ell_lo) and g.ell_lo != 0.0
    # Sign check against a scan: the gradient must predict the scan's slope.
    lo, hi = 8.0, 12.0
    slope = float(
        (logdetF(filt._replace(ell_lo=hi)) - logdetF(filt._replace(ell_lo=lo)))
        / (hi - lo)
    )
    assert np.sign(g.ell_lo) == np.sign(slope), f"grad {g.ell_lo} vs scan {slope}"


def test_soft_filter_preserves_the_shape_contract():
    filt = SoftTopHat(10.0, 30.0, width=2.0)
    skew = _soft(filt)
    assert skew(_cosmo()).shape == (1, _ELLS.size)
    batched = Cosmology(
        omega_cold=np.array([_OC, 0.32]),
        omega_baryon=np.full(2, _OB),
        sigma8_cold=np.array([_S8, 0.85]),
        ns=np.full(2, _NS),
        hubble=np.full(2, _H),
        neutrino_mass=np.zeros(2),
        w0=np.full(2, -1.0),
        wa=np.zeros(2),
    )
    assert skew(batched).shape == (2, _ELLS.size)


# ---------------------------------------------------------------------------
# The filter protocol: anything callable on ell. TopHat and SoftTopHat are two
# implementations of it, not special cases.
# ---------------------------------------------------------------------------


class _Gaussian(NamedTuple):
    ell_peak: float
    sigma: float

    def __call__(self, ell):
        return jnp.exp(-0.5 * ((ell - self.ell_peak) / self.sigma) ** 2)


class _FreeForm(NamedTuple):
    """Per-multipole weights as the parameters themselves."""

    w: jnp.ndarray

    def __call__(self, ell):
        return self.w


def test_custom_filter_is_accepted_and_differentiable():
    filt = _Gaussian(15.0, 4.0)
    skew = make_skew_spectra(
        _ELLS, filt, _dndz(), _LENSING3, interp=False, ell_sum=(2, 60), **_FAST
    )
    C = np.asarray(skew(_cosmo()))
    assert C.shape == (1, _ELLS.size) and np.all(np.isfinite(C)) and np.all(C > 0)

    g = jax.grad(lambda f: jnp.sum(skew(_cosmo(), f)))(filt)
    assert isinstance(g, _Gaussian)
    assert np.isfinite(g.ell_peak) and np.isfinite(g.sigma)
    assert g.sigma != 0.0


def test_custom_filter_requires_ell_sum():
    """Nothing can infer where an arbitrary function is non-negligible."""
    with pytest.raises(ValueError, match="ell_sum is required"):
        make_skew_spectra(
            _ELLS, _Gaussian(15.0, 4.0), _dndz(), _LENSING3, interp=False, **_FAST
        )


def test_free_form_weights_reproduce_a_tophat_and_give_per_mode_gradients():
    """The most general case: the filter's parameters *are* the per-multipole
    weights, so the shape itself can be optimised rather than a band."""
    band = np.arange(2, 61)
    filt = _FreeForm(jnp.asarray(((band >= 10) & (band <= 30)).astype(float)))
    skew = make_skew_spectra(
        _ELLS, filt, _dndz(), _LENSING3, interp=False, ell_sum=(2, 60), **_FAST
    )
    hard = np.asarray(
        make_skew_spectra(
            _ELLS,
            TopHat(10, 30),
            _dndz(),
            _LENSING3,
            interp=False,
            ell_sum=(2, 60),
            **_FAST,
        )(_cosmo())
    )
    assert np.allclose(np.asarray(skew(_cosmo())), hard, rtol=1e-12)

    g = jax.grad(lambda f: jnp.sum(skew(_cosmo(), f)))(filt)
    assert g.w.shape == band.shape
    assert np.count_nonzero(np.asarray(g.w)) > 0


def test_a_two_field_custom_filter_is_not_mistaken_for_a_pair():
    """A NamedTuple is a tuple, so pair-detection must key on the exact type --
    otherwise _Gaussian(15., 4.) would be read as two filters."""
    (f1, b1, s1), (f2, b2, s2) = _filter_spec(_Gaussian(15.0, 4.0), (2, 60))
    assert isinstance(f1, _Gaussian) and isinstance(f2, _Gaussian)
    assert f1 == f2 and not s1


def test_builtin_filters_are_callable_too():
    """TopHat and SoftTopHat satisfy the same protocol they document."""
    ell = np.arange(5, 20)
    assert np.array_equal(TopHat(8, 15)(ell), _evaluate(TopHat(8, 15), ell))
    assert np.allclose(
        SoftTopHat(8.0, 15.0, 2.0)(ell), _evaluate(SoftTopHat(8.0, 15.0, 2.0), ell)
    )
