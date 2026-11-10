"""The sigma8(z) that enters the BiHalofit coefficients: sigma8_input times the ratio of the top-hat sigma8 of the
spectrum at z to that at z = 0 (`bispectrum_all_z`).

The kernel does not return sigma8(z), so these tests act on it through B: they wrap the linear-spectrum backend and
check what the bispectrum does when the spectrum is changed in a way whose effect on sigma8(z) is known.
"""

import importlib

import jax
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from bihalofit_jax import make_bihalofit
from bihalofit_jax._pk_backends import _STANDARD_K_GRID, _W_SIGMA8, _pk_backend

bispectrum = importlib.import_module("bihalofit_jax.bispectrum")

K = np.array(
    [
        [0.05, 0.05, 0.05],
        [0.2, 0.2, 0.2],
        [1.0, 1.0, 1.0],
        [0.3, 0.3, 0.1],
        [2.0, 2.0, 0.5],
    ]
).T
Z = np.array([0.5, 1.0, 1.5])
A = 1.0 / (1.0 + Z)


@pytest.fixture
def patch_backend(monkeypatch):
    """patch(change): the 'eh' spectrum slices at a < 1 are replaced by change(k, PL_slices); z = 0 is left alone."""

    def patch(change):
        real = _pk_backend

        def backend(source, k_lin=None):
            fn = real(source, k_lin=k_lin)

            def wrapped(oc, ob, h, ns, s8, mnu, w0, wa, a, jd):
                PL, PL0 = fn(oc, ob, h, ns, s8, mnu, w0, wa, a, jd)
                k = np.asarray(_STANDARD_K_GRID if k_lin is None else k_lin)
                return change(k, PL, a), PL0

            return wrapped

        monkeypatch.setattr(bispectrum, "_pk_backend", backend)
        jax.clear_caches()  # the kernel is jitted on the backend's name, which has not changed

    yield patch
    monkeypatch.undo()
    jax.clear_caches()


def _bispectrum(params):
    return np.asarray(make_bihalofit(*K, A, pk_source="eh")(params))


def test_top_hat_weights_integrate_the_spectrum_to_sigma8(fiducial_params):
    """The quadrature weights give the top-hat sigma8 of a spectrum: checked against an adaptive integral of the
    cubic-spline interpolated E&H spectrum, which is normalised to sigma8 with the same weights.
    """
    _, PL0 = _pk_backend("eh")(
        *(
            fiducial_params[n]
            for n in (
                "omega_cold",
                "omega_baryon",
                "hubble",
                "ns",
                "sigma8_cold",
                "neutrino_mass",
                "w0",
                "wa",
            )
        ),
        np.array([1.0]),
        np.float64
    )
    pk = CubicSpline(np.log(_STANDARD_K_GRID), np.log(np.asarray(PL0)[0]))
    window = lambda x: 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    integral = quad(
        lambda lnk: np.exp(pk(lnk))
        * np.exp(3 * lnk)
        / (2 * np.pi**2)
        * window(8.0 * np.exp(lnk)) ** 2,
        np.log(_STANDARD_K_GRID[0]),
        np.log(_STANDARD_K_GRID[-1]),
        limit=500,
    )[0]
    assert np.sqrt(integral) == pytest.approx(
        fiducial_params["sigma8_cold"][0], rel=2e-3
    )
    assert np.sqrt(np.asarray(PL0)[0] @ _W_SIGMA8) == pytest.approx(
        fiducial_params["sigma8_cold"][0], rel=1e-10
    )


def test_sigma8_of_z_ignores_the_spectrum_at_scales_far_below_the_top_hat(
    fiducial_params, patch_backend
):
    """The top-hat is blind to k < 3e-3 h/Mpc, so a 5% change of the spectrum there, at z > 0 only, which a ratio of
    low-k power would read as a 2.5% change of growth, leaves B unchanged (to the 1e-4 that the change of
    the spectrum at the triangle's own k, here >= 0.05, allows)."""
    reference = _bispectrum(fiducial_params)
    patch_backend(
        lambda k, PL, a: PL
        * (1.0 - 0.05 * np.exp(-((k / 1e-3) ** 2))).astype(PL.dtype)[None, None, :]
    )
    assert np.abs(_bispectrum(fiducial_params) / reference - 1).max() < 1e-4


@pytest.mark.parametrize("factor", [0.8, 1.25])
def test_sigma8_of_z_follows_the_amplitude_of_the_spectrum_at_z(
    fiducial_params, patch_backend, factor
):
    """Multiplying the slices at z > 0 by factor^2 is, for E&H, the same as the spectrum of a cosmology with sigma8 times
    factor, whose growth is unchanged: both must give the same B, so sigma8(z) = sigma8 x factor x D(z) in the first.
    """
    scaled = {**fiducial_params, "sigma8_cold": fiducial_params["sigma8_cold"] * factor}
    expected = _bispectrum(scaled)
    patch_backend(lambda k, PL, a: PL * factor**2)
    np.testing.assert_allclose(_bispectrum(fiducial_params), expected, rtol=1e-10)
