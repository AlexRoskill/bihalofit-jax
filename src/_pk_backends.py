"""Linear P(k) backends, selected by ``pk_source``. All are JAX-differentiable.

  'syren'     (default) syren-new (Sui et al. 2025, arXiv:2410.14623), wCDM and neutrinos
  'baccoemu'  the baccoemu neural-network emulator (optional dependency)
  'eh'        Eisenstein & Hu 1998 no-wiggle formula
  'eh_99'     Eisenstein & Hu 1999 CDM+HDM formula at f_nu = 0, as in threepoint_py

Each returns PL_all (N_z, N_cosmo, N_k) and PL_z0 (N_cosmo, N_k) on its k grid; ``_pk_backend``
gives the uniform callable.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .background import growth_ratios as _growth_ratios

# Baccoemu's fixed linear P(k) k-grid in h/Mpc: 600 log-spaced points, 1e-4 to 50.
# Exact formula: no precision loss from string serialization.
# Must match emu.emulator['linear']['k']: verified at load time in _get_baccoemu().
_STANDARD_K_GRID = np.logspace(np.log10(1e-4), np.log10(50.0), 600)

# Linear P(k) backends accepted by pk_source (shared with projected_bihalofit).
_VALID_SOURCES = ("baccoemu", "eh", "eh_99", "syren")


# ---------------------------------------------------------------------------
# Quadrature weights on the k-grid (computed once at import)
# ---------------------------------------------------------------------------


def _trapezoid_dk(k):
    """Trapezoid quadrature weights dk for a 1-D grid (half-width endpoints).

    Parameters
    ----------
    k : (N_k,) ndarray, N_k >= 2

    Returns
    -------
    dk : (N_k,) ndarray
    """
    dk = np.empty(len(k))
    dk[0] = (k[1] - k[0]) / 2.0
    dk[1:-1] = (k[2:] - k[:-2]) / 2.0
    dk[-1] = (k[-1] - k[-2]) / 2.0
    return dk


def _build_sigma8_weights(k_grid):
    """Top-hat window (R=8 Mpc/h) quadrature weights for sigma8 normalisation.

    Parameters
    ----------
    k_grid : (N_k,) array-like
        Wavenumbers in h/Mpc.

    Returns
    -------
    w8 : (N_k,) float64 ndarray
        ``sigma8^2 = sum(w8 * P(k))``.
    """
    k = np.asarray(k_grid, dtype=np.float64)
    dk = _trapezoid_dk(k)
    x = k * 8.0  # k R, R=8 Mpc/h
    W_TH = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    return (dk * k**2 / (2.0 * np.pi**2) * W_TH**2).astype(np.float64)


_W_SIGMA8 = _build_sigma8_weights(_STANDARD_K_GRID)  # (N_k,)


# ---------------------------------------------------------------------------
# Eisenstein & Hu 1998 no-wiggle transfer function: shared by the 'eh'
# backend and the syren base spectrum
# ---------------------------------------------------------------------------


def _eh_nowiggle_T(k, Om_j, Ob_j, h_j):
    r"""EH98 no-wiggle transfer function T(k) (eqs 26, 28-31).

    k : (N_k,) in h/Mpc;  Om_j, Ob_j, h_j : (N_cosmo,) dimensionless.
    Returns T of shape (N_cosmo, N_k).

    The EH98 sound horizon s (eq 26) is in Mpc, so the dimensionless k*s
    product is k*h*s with k in h/Mpc.  The syren logF correction
    (_logF_fiducial_syren) was fitted against exactly this convention, so
    any change here must preserve the k*h*s ordering.
    """
    theta = jnp.asarray(2.7255 / 2.7, dtype=k.dtype)  # T_CMB / 2.7 K
    fb = Ob_j / Om_j  # baryon fraction
    om_h2 = Om_j * h_j**2  # Omega_m h^2
    ob_h2 = Ob_j * h_j**2  # Omega_b h^2

    # Sound horizon [Mpc] (eq 26)
    s = 44.5 * jnp.log(9.83 / om_h2) / jnp.sqrt(1.0 + 10.0 * ob_h2**0.75)

    # Shape-suppression factor (eq 31)
    alpha_G = (
        1.0 - 0.328 * jnp.log(431.0 * om_h2) * fb + 0.38 * jnp.log(22.3 * om_h2) * fb**2
    )

    # Effective shape parameter Gamma [h/Mpc] (eq 30), broadcast (N_cosmo, N_k)
    ks = k[None, :] * h_j[:, None] * s[:, None]  # k*s, dimensionless
    Gamma = (Om_j * h_j)[:, None] * (
        alpha_G[:, None] + (1.0 - alpha_G[:, None]) / (1.0 + (0.43 * ks) ** 4)
    )

    # Dimensionless wave number q and transfer function (eqs 28-29)
    q = k[None, :] * theta**2 / Gamma
    L0 = jnp.log(2.0 * jnp.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)  # (N_cosmo, N_k)


# ---------------------------------------------------------------------------
# Eisenstein & Hu 1998 P(k) backend: fully in JAX, differentiable
# ---------------------------------------------------------------------------


def _pk_lin_eh_jax(
    omega_m_j, omega_b_j, h_j, ns_j, sigma8_j, w0_j, wa_j, a_sorted, jdtype, k_lin=None
):
    """
    Linear P(k) from the Eisenstein & Hu 1998 no-wiggle fitting formula.

    The EH98 sound horizon s (eq 26) is in Mpc, so the k*s product uses
    k*h*s with k in h/Mpc (same convention as the syren backend).
    Neutrino mass is ignored (use the baccoemu backend for neutrino effects).

    Parameters
    ----------
    omega_m_j, omega_b_j, h_j, ns_j, sigma8_j, w0_j, wa_j : (N_cosmo,) JAX arrays
    a_sorted : (N_z,) ndarray
        Descending scale factors.
    jdtype : JAX dtype
    k_lin : array-like, optional
        Custom k-grid (h/Mpc); defaults to _STANDARD_K_GRID with
        precomputed sigma8 weights. A custom grid recomputes those weights.

    Returns
    -------
    PL_all : (N_z, N_cosmo, N_k) JAX array
        P(k, a) on the k-grid used.
    PL_z0 : (N_cosmo, N_k) JAX array
        P(k, a=1).
    """
    if k_lin is None:
        k = jnp.array(_STANDARD_K_GRID, dtype=jdtype)  # (N_k,)
        W8 = jnp.array(_W_SIGMA8, dtype=jdtype)  # (N_k,) sigma8 weights
    else:
        k = jnp.array(k_lin, dtype=jdtype)
        W8 = jnp.array(_build_sigma8_weights(np.asarray(k_lin)), dtype=jdtype)

    # --- EH98 no-wiggle transfer function (shared helper) ---
    T = _eh_nowiggle_T(k, omega_m_j, omega_b_j, h_j)  # (N_cosmo, N_k)

    # --- Unnormalised P(k, z=0) = k^ns * T^2 ---
    P_unnorm = k[None, :] ** ns_j[:, None] * T**2  # (N_cosmo, N_k)

    # --- Normalise to sigma8_cold ---
    sigma8_sq = jnp.einsum("ck,k->c", P_unnorm, W8)  # (N_cosmo,)
    P_z0 = P_unnorm * (sigma8_j / jnp.sqrt(sigma8_sq))[:, None] ** 2  # (N_cosmo, N_k)

    # --- Growth factor D(a)/D(1) via RK4 ODE solve (see background.growth_ratios) ---
    growths = _growth_ratios(a_sorted, omega_m_j, w0_j, wa_j, jdtype)  # (N_z, N_cosmo)

    # P(k, z) = P(k, z=0) * [D(a)/D(1)]^2
    PL_all = P_z0[None, :, :] * growths[:, :, None] ** 2  # (N_z, N_cosmo, N_k)

    return PL_all, P_z0


# ---------------------------------------------------------------------------
# Eisenstein & Hu 1999 CDM+HDM transfer function at f_nu = 0 (astro-ph/9710252), a different formula
# from 'eh' (EH98 no-wiggle; up to ~50% apart at low k). It matches threepoint_py's
# `cosmology.linear_pk` coefficient for coefficient, so the two codes can be compared on one P(k).
# ---------------------------------------------------------------------------


def _eh99_mixed_dm_T(k, Om_j, Ob_j, h_j):
    r"""Eisenstein & Hu 1999 CDM+HDM transfer function T(k), f_nu=0.

    k : (N_k,) in h/Mpc;  Om_j, Ob_j, h_j : (N_cosmo,) dimensionless.
    Returns T of shape (N_cosmo, N_k).

    Matches threepoint_py's cosmology.linear_pk exactly (same coefficients);
    k is converted to 1/Mpc internally (k*h), same convention as that code.
    """
    theta = jnp.asarray(2.728 / 2.7, dtype=k.dtype)  # T_CMB / 2.7 K
    fc = (Om_j - Ob_j) / Om_j  # cold (CDM) fraction; f_nu=0 => Om_c/Om_m
    fb = Ob_j / Om_j  # baryon fraction
    pc = 0.25 * (5.0 - jnp.sqrt(1.0 + 24.0 * fc))

    om_h2 = Om_j * h_j**2  # Omega_m h^2
    ob_h2 = Ob_j * h_j**2  # Omega_b h^2

    zeq = 2.5e4 * om_h2 / theta**4
    b1 = 0.313 * om_h2**-0.419 * (1.0 + 0.607 * om_h2**0.674)
    b2 = 0.238 * om_h2**0.223
    zd = 1291.0 * om_h2**0.251 / (1.0 + 0.659 * om_h2**0.828) * (1.0 + b1 * ob_h2**b2)
    yd = (1.0 + zeq) / (1.0 + zd)
    s = 44.5 * jnp.log(9.83 / om_h2) / jnp.sqrt(1.0 + 10.0 * ob_h2**0.75)  # Mpc

    alnu = (
        fc
        * (5.0 - 2.0 * pc)
        / 5.0
        * (1.0 - 0.553 * fb + 0.126 * fb**3)
        * (1.0 + yd) ** (-pc)
        * (1.0 + 0.5 * pc * (1.0 + 1.0 / (7.0 * (3.0 - 4.0 * pc))) / (1.0 + yd))
    )
    sqrt_alnu = jnp.sqrt(alnu)

    k_phys = k[None, :] * h_j[:, None]  # h/Mpc -> 1/Mpc
    geff = om_h2[:, None] * (
        sqrt_alnu[:, None]
        + (1.0 - sqrt_alnu[:, None]) / (1.0 + (0.43 * k_phys * s[:, None]) ** 4)
    )
    qeff = k_phys / geff * theta**2

    L = jnp.log(jnp.e + 1.84 * sqrt_alnu[:, None] * qeff / (1.0 - 0.949 * fb)[:, None])
    C = 14.4 + 325.0 / (1.0 + 60.5 * qeff**1.11)
    return L / (L + C * qeff**2)  # (N_cosmo, N_k)


def _pk_lin_eh99_jax(
    omega_m_j, omega_b_j, h_j, ns_j, sigma8_j, w0_j, wa_j, a_sorted, jdtype, k_lin=None
):
    """
    Linear P(k) from the Eisenstein & Hu 1999 CDM+HDM fitting formula at
    f_nu=0 (see _eh99_mixed_dm_T). Structurally identical to _pk_lin_eh_jax
    (same sigma8 normalisation and growth-factor machinery) -- only the
    transfer function differs; kept as a separate self-contained function
    (not sharing code with _pk_lin_eh_jax) to avoid touching that already
    validated/tested path.

    Neutrino mass is ignored (f_nu=0 throughout; use baccoemu for neutrino
    effects).

    Parameters
    ----------
    omega_m_j, omega_b_j, h_j, ns_j, sigma8_j, w0_j, wa_j : (N_cosmo,) JAX arrays
    a_sorted : (N_z,) ndarray
        Descending scale factors.
    jdtype : JAX dtype
    k_lin : array-like, optional
        Custom k-grid (h/Mpc); defaults to _STANDARD_K_GRID with
        precomputed sigma8 weights.

    Returns
    -------
    PL_all : (N_z, N_cosmo, N_k) JAX array
        P(k, a) on the k-grid used.
    PL_z0 : (N_cosmo, N_k) JAX array
        P(k, a=1).
    """
    if k_lin is None:
        k = jnp.array(_STANDARD_K_GRID, dtype=jdtype)  # (N_k,)
        W8 = jnp.array(_W_SIGMA8, dtype=jdtype)  # (N_k,) sigma8 weights
    else:
        k = jnp.array(k_lin, dtype=jdtype)
        W8 = jnp.array(_build_sigma8_weights(np.asarray(k_lin)), dtype=jdtype)

    # --- EH99 mixed-DM transfer function (f_nu=0) ---
    T = _eh99_mixed_dm_T(k, omega_m_j, omega_b_j, h_j)  # (N_cosmo, N_k)

    # --- Unnormalised P(k, z=0) = k^ns * T^2 ---
    # (threepoint_py's extra prefactors cancel in the sigma8 normalisation.)
    P_unnorm = k[None, :] ** ns_j[:, None] * T**2  # (N_cosmo, N_k)

    # --- Normalise to sigma8_cold ---
    sigma8_sq = jnp.einsum("ck,k->c", P_unnorm, W8)  # (N_cosmo,)
    P_z0 = P_unnorm * (sigma8_j / jnp.sqrt(sigma8_sq))[:, None] ** 2  # (N_cosmo, N_k)

    # --- Growth factor D(a)/D(1), as in _pk_lin_eh_jax ---
    growths = _growth_ratios(a_sorted, omega_m_j, w0_j, wa_j, jdtype)

    PL_all = P_z0[None, :, :] * growths[:, :, None] ** 2

    return PL_all, P_z0


# ---------------------------------------------------------------------------
# syren: symbolic_pofk plin_new_emulated (https://arxiv.org/abs/2410.14623)
# Translated to JAX from https://github.com/DeaglanBartlett/symbolic_pofk, which carries the following notice:
#
#   MIT License
#
#   Copyright (c) 2023 Deaglan Bartlett
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
# ---------------------------------------------------------------------------


def _sigma8_to_As_syren(sigma8_j, Om_j, Ob_j, h_j, ns_j, mnu_j, w0_j, wa_j, jdtype):
    """sigma8 → 10^9 As using the max-precision syren fitting formula.

    All cosmology inputs shape (N_cosmo,) (total Omega_m, Omega_b, h, n_s, sum m_nu in eV, w0, wa);
    jdtype sets the coefficient dtype. Returns 10^9 As, shape (N_cosmo,)."""
    # jdtype, not float64: a float64 coefficient array would silently promote everything
    # downstream (As, P_L, the k_NL solve) under x64, whatever dtype the caller asked for.
    b = jnp.array(
        [
            0.0246,
            2.1062,
            2.9355,
            0.7626,
            0.2962,
            0.5096,
            4.4025,
            3.6495,
            0.4144,
            0.8615,
            0.6188,
            0.1751,
            0.824,
            0.5466,
            0.5519,
            0.3689,
            0.3261,
            0.2002,
            0.8892,
            0.4462,
            1.215,
            3.4829,
            2.5852,
            0.0242,
            0.0051,
            0.1614,
            1.2991,
            4.1426,
            3.3055,
            0.5716,
            6.0094,
            1.9569,
            2.1477,
            1.1902,
            0.128,
            0.6931,
            0.2661,
        ],
        dtype=jdtype,
    )

    t1_inner = Om_j * b[1] + (
        b[2] * mnu_j - b[3] * ns_j + jnp.log(b[4] * h_j - b[5] * mnu_j)
    ) * (b[6] * h_j + b[7] * mnu_j - b[8] * ns_j + 1.0)
    t1 = b[0] * t1_inner

    t2 = b[9] * h_j - mnu_j

    t3_i1 = (b[12] * w0_j - b[13] * wa_j - jnp.log(Om_j * b[14])) * (
        Om_j * b[15]
        + b[16] * w0_j
        + b[17] * wa_j
        + jnp.log(-b[18] * w0_j - b[19] * wa_j)
    )
    t3_i2 = jnp.log(Om_j * b[20] + jnp.log(-b[21] * w0_j - b[22] * wa_j))
    t3 = (
        b[10] * w0_j
        - b[11] * mnu_j
        - t3_i1
        - t3_i2
        + jnp.log(-b[23] * w0_j - b[24] * wa_j)
    )

    t4_i1 = Ob_j * b[30] - b[31] * h_j - jnp.log(Om_j * b[32])
    t4_i2 = Om_j * b[33] - b[34] * h_j - b[35] * mnu_j - b[36] * ns_j
    t4 = (
        b[25] * mnu_j
        - jnp.sqrt(Ob_j) * b[26]
        - Ob_j * b[27]
        + Om_j * b[28]
        - b[29] * h_j
        + 1.0
        + t4_i1 * t4_i2
    )

    result = t1 * t2 * t3 * t4
    return (sigma8_j / result) ** 2  # 10^9 As


def _eh_nw_z0_syren(k, As_j, Om_j, Ob_j, h_j, ns_j):
    """E&H no-wiggle P(k) at z=0, taking As directly (no growth factor).
    k: (N_k,); cosmology: (N_cosmo,); returns (N_cosmo, N_k)."""
    T = _eh_nowiggle_T(k, Om_j, Ob_j, h_j)  # (N_cosmo, N_k)

    kpivot = 0.05
    pk = (
        2.0
        * jnp.pi**2
        / k[None, :] ** 3
        * (As_j[:, None] * 1e-9)
        * (k[None, :] * h_j[:, None] / kpivot) ** (ns_j[:, None] - 1.0)
        * (2.0 * k[None, :] ** 2 * 2998.0**2 / 5.0 / Om_j[:, None]) ** 2
        * T**2
    )
    return pk


def _logF_fiducial_syren(k, Om_j, Ob_j, h_j, jdtype):
    """ΛCDM transfer-function correction log(P_true / P_EH) from Bartlett+2024.
    k: (N_k,); cosmology: (N_cosmo,); returns (N_cosmo, N_k).
    Extrapolates outside the [9e-3, 9] h/Mpc tested range."""
    # jdtype, not a hardcoded float64 -- see _sigma8_to_As_syren's comment.
    b = jnp.array(
        [
            0.05448654,
            0.00379,
            0.0396711937097927,
            0.127733431568858,
            1.35,
            4.053543862744234,
            0.0008084539054750851,
            1.8852431049189666,
            0.11418372931475675,
            3.798,
            14.909,
            5.56,
            15.8274343004709,
            0.0230755621512691,
            0.86531976,
            0.8425442636372944,
            4.553956000000005,
            5.116999999999995,
            70.0234239999998,
            0.01107,
            5.35,
            6.421,
            134.309,
            5.324,
            21.532,
            4.741999999999985,
            16.68722499999999,
            3.078,
            16.987,
            0.05881491,
            0.0006864690561825617,
            195.498,
            0.0038454457516892,
            0.276696018851544,
            7.385,
            12.3960625361899,
            0.0134114370723638,
        ],
        dtype=jdtype,
    )

    # k: (N_k,) → broadcast with (N_cosmo,) cosmo params via [None,:] / [:,None]
    kk = k[None, :]  # (1, N_k)
    Om = Om_j[:, None]  # (N_cosmo, 1)
    Ob = Ob_j[:, None]
    h = h_j[:, None]

    line1 = b[0] * h - b[1]

    sqrt_h2_b3 = jnp.sqrt(h**2 + b[3])
    Ob_b2_over_sqrt = (Ob * b[2]) / sqrt_h2_b3
    line2 = Ob_b2_over_sqrt ** (b[4] * Om) * (
        (b[5] * kk - Ob)
        / jnp.sqrt(b[6] + (Ob - b[7] * kk) ** 2)
        * b[8]
        * (b[9] * kk) ** (-b[10] * kk)
        * jnp.cos(Om * b[11] - b[12] * kk / jnp.sqrt(b[13] + Ob**2))
        - b[14]
        * (b[15] * kk / jnp.sqrt(1.0 + b[16] * kk**2) - Om)
        * jnp.cos(b[17] * h / jnp.sqrt(1.0 + b[18] * kk**2))
    )

    line3 = (
        b[19]
        * (b[20] * Om + b[21] * h - jnp.log(b[22] * kk) + (b[23] * kk) ** (-b[24] * kk))
        * jnp.cos(b[25] / jnp.sqrt(1.0 + b[26] * kk**2))
    )

    line4 = (
        (b[27] * kk) ** (-b[28] * kk)
        * (
            b[29] * kk
            - (b[30] * jnp.log(b[31] * kk)) / jnp.sqrt(b[32] + (Om - b[33] * h) ** 2)
        )
        * jnp.cos(Om * b[34] - b[35] * kk / jnp.sqrt(Ob**2 + b[36]))
    )

    return line1 + line2 + line3 + line4  # (N_cosmo, N_k)


def _log10_S_syren(k, Om_j, Ob_j, h_j, mnu_j, w0_j, wa_j, jdtype):
    """z=0 power-spectrum shape correction for wCDM/mnu from linear_new.py.
    k: (N_k,); cosmology: (N_cosmo,); returns (N_cosmo, N_k)."""
    # jdtype, not a hardcoded float64 -- see _sigma8_to_As_syren's comment.
    e = jnp.array(
        [
            0.2841,
            0.1679,
            0.0534,
            0.0024,
            0.1183,
            0.3971,
            0.0985,
            0.0009,
            0.1258,
            0.2476,
            0.1841,
            0.0316,
            0.1385,
            0.2825,
            0.8098,
            0.019,
            0.1376,
            0.3733,
        ],
        dtype=jdtype,
    )

    kk = k[None, :]
    Om = Om_j[:, None]
    Ob = Ob_j[:, None]
    h = h_j[:, None]
    mnu = mnu_j[:, None]
    w0 = w0_j[:, None]
    wa = wa_j[:, None]

    p1 = -e[0] * h
    p2 = -e[1] * w0
    p3 = -e[2] * mnu / jnp.sqrt(e[3] + kk**2)
    p4 = -(e[4] * h) / (e[5] * h + mnu)
    p5 = e[6] * mnu / (h * jnp.sqrt(e[7] + (Om * e[8] + kk) ** 2))

    numer = (
        e[9] * Ob - e[10] * w0 - e[11] * wa + (e[12] * w0 + e[13]) / (e[14] * wa + w0)
    )
    denom = jnp.sqrt(e[15] + (Om + e[16] * jnp.log(-e[17] * w0)) ** 2)
    p6 = numer / denom

    return (p1 + p2 + p3 + p4 + p5 + p6) / 10.0  # (N_cosmo, N_k)


def _growth_R_syren(Om_j, Ob_j, h_j, mnu_j, w0_j, wa_j, a, jdtype):
    """Growth-factor wCDM correction from linear_new.py.

    Cosmology inputs: (N_cosmo,); a: scalar scale factor (may be a tracer); jdtype: coefficient
    dtype. Ob_j and h_j are unused. Returns the multiplicative correction, (N_cosmo,).
    """
    # jdtype, not a hardcoded float64 -- see _sigma8_to_As_syren's comment.
    d = jnp.array(
        [
            0.8545,
            0.394,
            0.7294,
            0.5347,
            0.4662,
            4.6669,
            0.4136,
            1.4769,
            0.5959,
            0.4553,
            0.0799,
            5.8311,
            5.8014,
            6.7085,
            0.3445,
            1.2498,
            0.3756,
            0.2136,
        ],
        dtype=jdtype,
    )

    denom1 = (
        a * d[1] + d[2] + (Om_j * d[3] - a * d[4]) * jnp.log(-d[5] * w0_j - d[6] * wa_j)
    )
    part2 = -1.0 / denom1

    numer3 = Om_j * d[7] - a * d[8] + jnp.log(-d[9] * w0_j - d[10] * wa_j)
    denom3 = (
        -a * d[11]
        + d[12]
        + d[13] * (Om_j * d[14] + a * d[15] - 1.0) * (d[16] * w0_j + d[17] * wa_j + 1.0)
    )
    part3 = -numer3 / denom3

    return 1.0 + (1.0 - a) * (d[0] + part2 + part3)


def _growth_D_syren(k, Om_j, Ob_j, h_j, mnu_j, w0_j, wa_j, a):
    """k-dependent growth factor (Bond+1980 + wCDM) from linear_new.py.
    k: (N_k,); cosmology: (N_cosmo,); a: Python float.
    Returns (N_cosmo, N_k)."""
    # Avoid neutrino singularity
    mnu_j = mnu_j + 1e-10

    theta = 2.7255 / 2.7
    z = 1.0 / a - 1.0
    om_h2 = Om_j * h_j**2
    zeq = 2.5e4 * om_h2 / theta**4  # (N_cosmo,)

    # wCDM Omega(a) and Omega_Lambda(a) for growth
    Om_a = Om_j * a ** (-3)
    OL_a = (
        (1.0 - Om_j)
        * a ** (-3 * (1.0 + w0_j + wa_j))
        * jnp.exp(-3.0 * wa_j * (1.0 - a))
    )
    g2 = Om_a + OL_a
    Om_n = Om_a / g2
    OL_n = OL_a / g2

    # Carroll+1992 / Lahav+1992 growth factor (no k-dependence yet)
    D1 = (
        (1.0 + zeq)
        / (1.0 + z)
        * 5.0
        * Om_n
        / 2.0
        / (Om_n ** (4.0 / 7.0) - OL_n + (1.0 + Om_n / 2.0) * (1.0 + OL_n / 70.0))
    )

    # Neutrino fraction; fcb = fc + fb = 1 - fnu
    Onu = mnu_j / 93.14 / h_j**2
    fnu = Onu / Om_j
    Nnu = 3.0  # assume 3 degenerate species when mnu > 0

    # Bond+1980 neutrino suppression, k-dependent:
    #   Dcbnu = (fcb**(0.7/pcb) + (D1/(1+yfs))**0.7)**(pcb/0.7) * D1**(1-pcb)
    # Evaluated in log space with pcb in cancellation-free form: as fnu -> 0, yfs ~ fnu**(-13/6)
    # overflows float32 in the backward pass and pcb cancels to 0.
    pcb = (6.0 * fnu / (5.0 + jnp.sqrt(25.0 - 24.0 * fnu)))[:, None]
    log_fnu = jnp.log(fnu)[:, None]
    log_D1 = jnp.log(D1)[:, None]
    q = k[None, :] * h_j[:, None] * theta**2 / om_h2[:, None]  # (N_cosmo, N_k)
    log_yfs = (
        jnp.log(17.2)
        + log_fnu
        + jnp.logaddexp(0.0, jnp.log(0.488) - 7.0 / 6.0 * log_fnu)
        + 2.0 * (jnp.log(Nnu * q) - log_fnu)
    )
    log_A = 0.7 * jnp.log1p(-fnu)[:, None] / pcb  # log fcb**(0.7/pcb)
    log_T = 0.7 * (log_D1 - jnp.logaddexp(0.0, log_yfs))  # log (D1/(1+yfs))**0.7
    Dcbnu = jnp.exp(pcb / 0.7 * jnp.logaddexp(log_A, log_T) + (1.0 - pcb) * log_D1)

    # Remove (1+zeq) normalisation from E&H 1997 convention
    return Dcbnu / (1.0 + zeq[:, None])


def _pk_lin_syren_jax(
    omega_cold_j,
    omega_baryon_j,
    hubble_j,
    ns_j,
    sigma8_cold_j,
    neutrino_mass_j,
    w0_j,
    wa_j,
    a_sorted,
    jdtype,
    k_lin=None,
):
    """
    Linear P(k) from symbolic_pofk plin_new_emulated (Bartlett et al. 2023/2024).

    Fully JAX: no external imports beyond numpy constants.
    Supports wCDM (w0, wa) and neutrino mass via analytical fitting formulas.

    The amplitude follows the other backends: syren's sigma8 -> As formula (_sigma8_to_As_syren) is a fit, good
    to ~0.1% in sigma8, so the spectrum is rescaled to have exactly the requested top-hat sigma8.

    Parameters
    ----------
    omega_cold_j, omega_baryon_j, hubble_j, ns_j, sigma8_cold_j, neutrino_mass_j, w0_j, wa_j : (N_cosmo,) JAX arrays
        ``neutrino_mass_j`` in eV.
    a_sorted : (N_z,) array
        Scale factors, numpy or a JAX tracer.
    jdtype : JAX dtype
    k_lin : array-like, optional
        Custom k-grid (h/Mpc); defaults to _STANDARD_K_GRID, with the sigma8
        weights recomputed for a custom grid, as in the eh backend.

    Returns
    -------
    PL_all : (N_z, N_cosmo, N_k) JAX array
        P(k, a) on the k-grid used.
    PL_z0 : (N_cosmo, N_k) JAX array
        P(k, a=1).
    """
    k = jnp.array(_STANDARD_K_GRID if k_lin is None else k_lin, dtype=jdtype)  # (N_k,)

    # Total matter density including neutrinos.
    # omega_cold_j = omega_cdm + omega_baryon in this codebase: do NOT add omega_baryon again.
    Onu_j = neutrino_mass_j / 93.14 / hubble_j**2
    Om_j = omega_cold_j + Onu_j  # (N_cosmo,)

    # sigma8 → As via syren's own fitting formula
    As_j = _sigma8_to_As_syren(
        sigma8_cold_j,
        Om_j,
        omega_baryon_j,
        hubble_j,
        ns_j,
        neutrino_mass_j,
        w0_j,
        wa_j,
        jdtype,
    )

    # Time-independent base spectrum: E&H no-wiggle × logF × S
    pk_eh = _eh_nw_z0_syren(k, As_j, Om_j, omega_baryon_j, hubble_j, ns_j)
    logF = _logF_fiducial_syren(k, Om_j, omega_baryon_j, hubble_j, jdtype)
    log10S = _log10_S_syren(
        k, Om_j, omega_baryon_j, hubble_j, neutrino_mass_j, w0_j, wa_j, jdtype
    )
    pk_base = pk_eh * jnp.exp(logF) * jnp.power(10.0, log10S)  # (N_cosmo, N_k)

    # z=0 slice: P(k, a=1)
    D_z0 = _growth_D_syren(
        k, Om_j, omega_baryon_j, hubble_j, neutrino_mass_j, w0_j, wa_j, 1.0
    )  # (N_cosmo, N_k)
    R_z0 = _growth_R_syren(
        Om_j, omega_baryon_j, hubble_j, neutrino_mass_j, w0_j, wa_j, 1.0, jdtype
    )[
        :, None
    ]  # (N_cosmo, 1)
    PL_z0 = pk_base * D_z0**2 * R_z0  # (N_cosmo, N_k)

    # Redshift slices: vmap over the scale factor so a_sorted may be a JAX tracer, as in
    # the Limber projection, where a(chi) depends on cosmology and must stay differentiable.
    a_arr = jnp.asarray(a_sorted, dtype=jdtype)  # (N_z,)

    def _slice(a):
        """P(k) at scalar scale factor a, (N_cosmo, N_k)."""
        D_a = _growth_D_syren(
            k, Om_j, omega_baryon_j, hubble_j, neutrino_mass_j, w0_j, wa_j, a
        )  # (N_cosmo, N_k)
        R_a = _growth_R_syren(
            Om_j, omega_baryon_j, hubble_j, neutrino_mass_j, w0_j, wa_j, a, jdtype
        )[:, None]
        return pk_base * D_a**2 * R_a

    PL_all = jax.vmap(_slice)(a_arr)  # (N_z, N_cosmo, N_k)

    # sigma8 normalisation, as for eh and baccoemu: the spectrum's own top-hat sigma8 is set to the requested one
    W8 = jnp.array(
        _W_SIGMA8 if k_lin is None else _build_sigma8_weights(np.asarray(k_lin)),
        dtype=jdtype,
    )
    scale = sigma8_cold_j**2 / jnp.einsum("ck,k->c", PL_z0, W8)  # (N_cosmo,)
    return PL_all * scale[None, :, None], PL_z0 * scale[:, None]


# ---------------------------------------------------------------------------
# baccoemu
# ---------------------------------------------------------------------------

_baccoemu_instance = None


def _get_baccoemu():
    """The shared baccoemu ``Matter_powerspectrum`` instance, loaded on first call.

    Takes no arguments. Raises ImportError if baccoemu is not installed, RuntimeError if its
    linear k-grid differs from ``_STANDARD_K_GRID``.
    """
    global _baccoemu_instance
    if _baccoemu_instance is None:
        try:
            from baccoemu import Matter_powerspectrum
        except ImportError as e:
            raise ImportError(
                "pk_source='baccoemu' requires the optional dependency "
                "'baccoemu'. Install it with:\n"
                "    pip install 'bihalofit-jax[baccoemu]'\n"
                "or use pk_source='syren' or pk_source='eh' (no extra "
                "dependencies)."
            ) from e
        _baccoemu_instance = Matter_powerspectrum()
        _k_emu = np.asarray(_baccoemu_instance.emulator["linear"]["k"])
        if _k_emu.shape != _STANDARD_K_GRID.shape or not np.allclose(
            _k_emu, _STANDARD_K_GRID, rtol=1e-10
        ):
            raise RuntimeError(
                "The installed baccoemu's linear k-grid does not match the "
                "grid assumed by bihalofit_jax (600 log-spaced points, "
                "1e-4 to 50 h/Mpc). This baccoemu version is incompatible."
            )
    return _baccoemu_instance


def _validate_baccoemu_bounds(p_numpy, a_sorted=None):
    """Check inputs against the emulator's trained parameter bounds.

    Calling model_cold directly bypasses baccoemu's own validation, so
    out-of-prior inputs would silently extrapolate the network.

    p_numpy must be plain numpy arrays (not JAX arrays) so validation runs
    before any H2D transfer and avoids forced D2H syncs. JAX tracers (inside
    jacfwd) are skipped: the concrete primal call has already validated them.

    a_sorted=None skips the expfactor check (the Limber path, where the
    scale factors are cosmology-dependent tracers checked at setup instead).

    Parameters
    ----------
    p_numpy : dict
        Maps 'omega_cold', 'omega_baryon', 'hubble', 'neutrino_mass', 'w0', 'wa' (any subset) to
        arrays of values.
    a_sorted : (N_z,) array-like, optional
        Scale factors.

    Returns
    -------
    None
        Raises ValueError on an out-of-bounds value.
    """
    lin = _get_baccoemu().emulator["linear"]
    bounds = dict(zip(lin["keys"], np.asarray(lin["bounds"])))
    values = {
        k: p_numpy[k]
        for k in ("omega_cold", "omega_baryon", "hubble", "neutrino_mass", "w0", "wa")
        if k in p_numpy
    }
    if a_sorted is not None:
        values["expfactor"] = a_sorted
    for name, v in values.items():
        try:
            v = np.asarray(v)
        except jax.errors.TracerArrayConversionError:
            continue
        lo, hi = bounds[name]
        if np.any(v < lo) or np.any(v > hi):
            raise ValueError(
                f"{name}={v} is outside the baccoemu emulator bounds "
                f"[{lo}, {hi}]; the emulator would extrapolate silently. "
                "Use pk_source='syren' or 'eh' for wider parameter ranges."
            )


def _pk_lin_baccoemu_jax(
    omega_cold_j,
    omega_baryon_j,
    hubble_j,
    ns_j,
    sigma8_cold_j,
    neutrino_mass_j,
    w0_j,
    wa_j,
    a_sorted,
    jdtype,
):
    """
    Linear cold-matter P(k) from baccoemu, fully JAX-traceable.

    Stacks all N_cosmo × (N_z + 1) inputs into a single (N_cosmo*(N_z+1), 7)
    matrix and calls model_cold once, matching the batching efficiency of the
    numpy path while remaining differentiable via jacfwd.

    Parameters
    ----------
    omega_cold_j, omega_baryon_j, hubble_j, ns_j, sigma8_cold_j, neutrino_mass_j, w0_j, wa_j : (N_cosmo,) JAX arrays
        In dtype ``jdtype``.
    a_sorted : (N_z,) ndarray
        Descending scale factors.
    jdtype : JAX dtype

    Returns
    -------
    PL_all : (N_z, N_cosmo, N_k) JAX array
        P(k, a) on _STANDARD_K_GRID.
    PL_z0 : (N_cosmo, N_k) JAX array
        P(k, a=1).
    """
    lin = _get_baccoemu().emulator["linear"]
    model = lin["model_cold"]
    scaler = lin["scaler_cold"]
    kk = jnp.array(lin["k"], dtype=jdtype)  # (N_k,)
    W8 = jnp.array(_W_SIGMA8, dtype=jdtype)  # (N_k,)

    N_cosmo = omega_cold_j.shape[0]
    # Append a=1 for the sigma8 normalisation; jnp so a_sorted may be a tracer (Limber path).
    a_all = jnp.concatenate(
        [jnp.asarray(a_sorted, dtype=jdtype), jnp.ones((1,), dtype=jdtype)]
    )  # (N_z+1,)
    n_slices = a_all.shape[0]

    # (N_cosmo * n_slices, 7) emulator input; ns is not an input (trained at ns=1).
    a_tiled = jnp.repeat(a_all, N_cosmo)  # (N_cosmo*n_slices,)

    def tile(v):
        """(N_cosmo,) -> (N_cosmo*n_slices,), one copy per scale factor."""
        return jnp.tile(v, n_slices)

    X = jnp.stack(
        [
            tile(omega_cold_j),
            tile(omega_baryon_j),
            tile(hubble_j),
            tile(neutrino_mass_j),
            tile(w0_j),
            tile(wa_j),
            a_tiled,
        ],
        axis=1,
    )

    # Single batched forward pass
    ypred = jnp.exp(
        scaler.inverse_transform(model(X)).astype(jdtype)
    )  # (N_cosmo*n_slices, N_k)

    # ns tilt (k h / 0.05)^(ns - 1)
    h_tiled = tile(hubble_j)[:, None]  # (N_cosmo*n_slices, 1)
    ns_tiled = tile(ns_j)[:, None]
    ypred *= (kk[None, :] * h_tiled / 0.05) ** (ns_tiled - 1.0)

    # Reshape to (n_slices, N_cosmo, N_k)
    ypred = ypred.reshape(n_slices, N_cosmo, -1)

    # sigma8 normalisation from the appended z=0 slice
    pk_z0 = ypred[-1]  # (N_cosmo, N_k)
    scale = (sigma8_cold_j / jnp.sqrt(jnp.einsum("ck,k->c", pk_z0, W8))) ** 2

    PL_z0 = pk_z0 * scale[:, None]
    PL_all = ypred[:-1] * scale[None, :, None]  # (N_z, N_cosmo, N_k)
    return PL_all, PL_z0


# ---------------------------------------------------------------------------
# Lookup on the k-grid (host-side, once per build)
# ---------------------------------------------------------------------------


def _pk_log_interp_indices(k_arr, jdtype, k_lin=None):
    """Left-grid index and log-linear interpolation weight of k_arr against
    a k-grid (defaults to the standard P(k) grid).

    Parameters
    ----------
    k_arr : array-like, any shape
        Wavenumbers in h/Mpc.
    jdtype : JAX dtype
        Sets the dtype of ``w``.
    k_lin : (N_k,) array-like, optional
        P(k) grid in h/Mpc (default ``_STANDARD_K_GRID``).

    Returns
    -------
    idx : int32 ndarray, shape of k_arr
        Left grid index, clipped to [0, N_k-2].
    w : ndarray, shape of k_arr
        Weight of the right neighbour in log k.
    """
    if k_lin is None:
        k_lin = _STANDARD_K_GRID
    log_k_lin = np.log(k_lin)
    N_k = len(k_lin)
    fpdtype = np.float64 if jdtype == jnp.float64 else np.float32

    k_arr = np.asarray(k_arr)
    lk = np.log(k_arr)
    flat_idx = np.clip(
        np.searchsorted(log_k_lin, lk.ravel(), side="right") - 1, 0, N_k - 2
    )
    idx = flat_idx.reshape(k_arr.shape)
    w = ((lk - log_k_lin[idx]) / (log_k_lin[idx + 1] - log_k_lin[idx])).astype(fpdtype)
    return idx.astype(np.int32), w


def _pk_backend(pk_source, k_lin=None):
    """Return a uniform P(k) callable fn(oc, ob, h, ns, s8, mnu, w0, wa, a, jdtype).

    Dispatches to one of the four linear P(k) backends. All return (PL_all, PL_z0) with
    PL_all of shape (len(a), N_cosmo, N_k) and PL_z0 of shape (N_cosmo, N_k); 'eh' and
    'eh_99' ignore neutrino mass.

    The callable's cosmology arguments are (N_cosmo,) arrays; a is (N_z,) scale factors and
    jdtype the JAX dtype.

    Parameters
    ----------
    pk_source : {'baccoemu', 'syren', 'eh', 'eh_99'}
    k_lin : array-like, optional
        Custom k-grid (h/Mpc), forwarded to eh/eh_99/syren. baccoemu cannot
        take one (its output grid is fixed by the trained emulator), so it raises.
    """
    if pk_source == "syren":
        return lambda oc, ob, h, ns, s8, mnu, w0, wa, a, jd: _pk_lin_syren_jax(
            oc, ob, h, ns, s8, mnu, w0, wa, a, jd, k_lin=k_lin
        )
    if pk_source == "baccoemu":
        if k_lin is not None:
            raise ValueError(
                "baccoemu does not support a custom k_lin "
                "(fixed native emulator grid)"
            )
        return lambda oc, ob, h, ns, s8, mnu, w0, wa, a, jd: _pk_lin_baccoemu_jax(
            oc, ob, h, ns, s8, mnu, w0, wa, a, jd
        )
    if pk_source == "eh":
        return lambda oc, ob, h, ns, s8, mnu, w0, wa, a, jd: _pk_lin_eh_jax(
            oc, ob, h, ns, s8, w0, wa, a, jd, k_lin=k_lin
        )
    if pk_source == "eh_99":
        return lambda oc, ob, h, ns, s8, mnu, w0, wa, a, jd: _pk_lin_eh99_jax(
            oc, ob, h, ns, s8, w0, wa, a, jd, k_lin=k_lin
        )
    raise ValueError(f"pk_source must be one of {_VALID_SOURCES} (got {pk_source!r}).")
