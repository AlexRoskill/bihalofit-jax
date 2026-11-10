r"""Skew spectra :math:`C_\ell^{(jk)i}` from the Limber-projected bispectrum, one tracer per leg:

.. math::
    C_{\ell}^{(jk)i} = \sum_{\ell_1 \ell_2}
        \frac{(2\ell_1+1)(2\ell_2+1)}{4\pi}
        \begin{pmatrix} \ell_1 & \ell_2 & \ell \\ m_1 & m_2 & m \end{pmatrix}
        \begin{pmatrix} \ell_1 & \ell_2 & \ell \\ m_1' & m_2' & m' \end{pmatrix}
        F_{\ell_1} F_{\ell_2}\, B_{ijk}(\ell, \ell_1, \ell_2)

The two 3j symbols may have different :math:`m`, which covers spin-:math:`s` fields; ``(0,0,0)``
for both with top-hat filters is the filtered-squared-field estimator of ``anicola/bispectrum``.
Legs: ``filters[0]`` on leg 1 (summed over l1), ``filters[1]`` on leg 2 (over l2), leg 3 unfiltered at l.

Everything cosmology-independent is folded into one sparse weight :math:`W[\ell,\ell_1,\ell_2]`, so a
call evaluates :math:`B`, multiplies and scatter-adds. 3j symbols come from the optional ``pyshtools``
and are cached on disk. Multipoles are integers.
"""

import hashlib
import os
import warnings

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from ._config import resolve_dtype
from .projected_bispectrum import (
    make_projected_bihalofit,
    _CHI_MIN_DEFAULT,
    _CHI_MAX_DEFAULT,
)

# precompute_w3j allocates one dense float64 (n_ell, n_1, n_2) block. Warn past
# this size: the reference implementation's unrestricted (lmax, lmax, lmax)
# cube is 8 GB at lmax = 1000, which restricting to the filter support avoids.
_W3J_WARN_GIB = 1.0

# 'auto' spacing uses the reference implementation's hybrid set below this
# l_min, where a linear grid cannot resolve the fast low-ell variation of B.
_HYBRID_BELOW = 10

# Cap on the auto-derived node count. The grid is a cube, so the
# projected-bispectrum call allocates n**3 triangles; past ~150 nodes at
# N_chi=160/float64 that exhausts GPU memory.
_N_NODES_MAX = 140

# Bumped whenever a change would alter precompute_w3j's output, so stale
# cache files can never be read back as valid.
_W3J_CACHE_VERSION = 1


class TopHat(NamedTuple):
    r"""Hard band filter: :math:`F_\ell = 1` on ``[ell_lo, ell_hi]``
    inclusive, 0 outside.

    Static and fast. **Its edges cannot be differentiated**: multipoles are
    integers, so the derivative is zero almost everywhere and undefined on
    them. Use :class:`SoftTopHat` to vary a band.
    """

    ell_lo: float
    ell_hi: float

    def __call__(self, ell):
        """Filter at multipoles ``ell`` (array): 1.0 inside the band, 0.0 outside, shape of ell."""
        return ((ell >= self.ell_lo) & (ell <= self.ell_hi)).astype(float)


class SoftTopHat(NamedTuple):
    r"""Smooth band filter: ~1 on ``[ell_lo, ell_hi]``, tapering to 0 over
    ``width`` multipoles at each edge with a logistic ramp.

    A pytree, so gradients come back named (``g.ell_lo``). Logistic rather
    than a clipped ramp because it is C-infinity, so second derivatives stay
    unambiguous. Converges to :class:`TopHat` as ``width -> 0``; at finite
    ``width`` the band really is softened. Keep ``width`` >~ 1: the sum
    samples the filter only at integer multipoles, so a narrower ramp is
    unresolved and its gradient unreliable.

    """

    ell_lo: float
    ell_hi: float
    width: float = 1.0

    def __call__(self, ell):
        """Filter at multipoles ``ell`` (array), in (0, 1), shape of ell."""
        # Ramps centred half a multipole outside the band, so that ell_lo and
        # ell_hi are fully *included* and width -> 0 reproduces TopHat exactly.
        # Centring on the edge itself would give the boundary multipoles
        # weight 1/2.
        return jax.nn.sigmoid(
            (ell - (self.ell_lo - 0.5)) / self.width
        ) * jax.nn.sigmoid(((self.ell_hi + 0.5) - ell) / self.width)


# Where a SoftTopHat's support is truncated when ell_sum is not given, in
# units of `width`. sigmoid(-8) ~ 3e-4, small enough to ignore.
_SOFT_SUPPORT_WIDTHS = 8


def _evaluate(filt, ell):
    """Filter values at the multipoles ``ell``.

    Any callable is called directly -- that is the whole filter protocol.
    :class:`TopHat` and :class:`SoftTopHat` are simply two implementations of
    it; a raw array is indexed instead.

    Parameters
    ----------
    filt : callable or 1-D array-like
        Filter; an array is indexed by integer multipole from 0.
    ell : array
        Multipoles.

    Returns
    -------
    F : shape of ell
    """
    if callable(filt):
        return filt(ell)
    return np.asarray(filt)[np.asarray(ell, dtype=int)]


def _filter_spec(filters, ell_sum):
    """Normalise the ``filters`` argument to ``(spec1, spec2)``.

    Each spec is ``(filt_or_array, band, is_static)`` where ``band`` is the
    fixed integer support of that leg's sum. Accepts one filter for both legs
    or a 2-tuple, and raw arrays indexed from ell = 0 for arbitrary shapes.

    Parameters
    ----------
    filters : filter or 2-tuple of filters
        As in :func:`make_skew_spectra`.
    ell_sum : (int, int) or None
        Summed multipole range; required for custom callables.

    Returns
    -------
    specs : list of two ``(filt_or_array, band, is_static)`` tuples
        ``band`` is an int ndarray of the multipoles summed over.
    """
    # A NamedTuple filter is itself a tuple, so only a *plain* tuple counts as
    # a per-leg pair (a two-field custom filter would otherwise be read as two).
    if type(filters) is not tuple:
        filters = (filters, filters)
    if len(filters) != 2:
        raise ValueError(
            f"filters must be one filter (used on both summed legs) or a "
            f"2-tuple, one per leg; got {len(filters)} entries."
        )

    specs = []
    for i, f in enumerate(filters):
        if isinstance(f, TopHat):
            static, span = True, (int(np.ceil(f.ell_lo)), int(np.floor(f.ell_hi)))
        elif isinstance(f, SoftTopHat):
            m = int(np.ceil(_SOFT_SUPPORT_WIDTHS * float(f.width)))
            static = False
            span = (int(np.floor(f.ell_lo)) - m, int(np.ceil(f.ell_hi)) + m)
        elif callable(f):
            # A user-supplied filter: arbitrary function of ell, differentiable
            # in whatever parameters it carries. Nothing here can infer where
            # it is non-negligible, so the caller must say.
            if ell_sum is None:
                raise ValueError(
                    f"filters[{i}] is a custom callable, so ell_sum is "
                    f"required: the range of ell1, ell2 summed over cannot be "
                    f"inferred from an arbitrary function. Pass "
                    f"ell_sum=(lo, hi) covering where the filter is "
                    f"appreciably non-zero."
                )
            static, span = False, (int(ell_sum[0]), int(ell_sum[1]))
        else:
            arr = np.asarray(f, dtype=float)
            if arr.ndim != 1:
                raise ValueError(
                    f"filters[{i}] must be a filter object, a callable of "
                    f"ell, or a 1-D array indexed by integer multipole from "
                    f"0; got shape {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"filters[{i}] contains non-finite entries.")
            band = np.nonzero(arr)[0]
            if band.size == 0:
                raise ValueError(f"filters[{i}] is identically zero.")
            if band[0] < 1:
                raise ValueError(
                    "Filters must vanish at ell = 0; the sum is log-spaced " "in ell."
                )
            specs.append((arr, band, True))
            continue

        lo, hi = ell_sum if ell_sum is not None else span
        lo = max(int(lo), 1)
        if hi < lo:
            raise ValueError(f"filters[{i}] has empty support.")
        specs.append((f, np.arange(lo, int(hi) + 1), static))
    return specs


def _w3j_cache_path(cache_dir, ells, ell1, ell2, m):
    """Content-addressed path for one 3j block: everything that changes the
    result is in the digest, so a stale hit is impossible.

    Parameters
    ----------
    cache_dir : str
    ells, ell1, ell2 : int array-like
        Multipoles of the block.
    m : tuple of 3 int

    Returns
    -------
    path : str
        ``<cache_dir>/w3j_<hash>.npy``.
    """
    h = hashlib.sha256()
    h.update(np.asarray((_W3J_CACHE_VERSION,) + tuple(m), dtype=np.int64).tobytes())
    for a in (ells, ell1, ell2):
        h.update(np.asarray(a, dtype=np.int64).tobytes())
    return os.path.join(cache_dir, f"w3j_{h.hexdigest()[:32]}.npy")


def precompute_w3j(ells, ell1, ell2, m=(0, 0, 0), cache_dir=None):
    r"""Wigner-3j symbols :math:`(\ell_1\, \ell_2\, \ell;\, m_1\, m_2\,
    m_3)` on the block ``ells x ell1 x ell2``, as ``(n_ell, n_1, n_2)``.

    ``m`` attaches to :math:`(\ell_1, \ell_2, \ell)` and must sum to zero.
    The result is **not** squared -- the caller squares it by asking for the
    same ``m`` twice. ``ell1``/``ell2`` are the filter supports, not the full
    range: the block is dense, so restricting them is what keeps this off an
    :math:`\ell_\mathrm{max}^3` cube.

    ``cache_dir`` memoises the block as a ``.npy`` named by a hash of
    ``(ells, ell1, ell2, m)`` -- content-addressed, so it cannot go stale. Off
    by default (blocks reach several GiB); worth it for wide bands, where this
    dominates the build (10.4 s for a 20-1000 band against ~1.3 s to read
    back). Needs the optional ``pyshtools`` dependency, unless cached.

    Parameters
    ----------
    ells, ell1, ell2 : int array-like
        Non-negative multipoles; ``ells`` is :math:`\ell`, the others :math:`\ell_1, \ell_2`.
    m : tuple of 3 int, optional
        :math:`(m_1, m_2, m_3)`, summing to zero.
    cache_dir : str or None, optional

    Returns
    -------
    w3j : (n_ell, n_1, n_2) float64 ndarray
        Memory-mapped when read from the cache.
    """
    m1, m2, m3 = (int(x) for x in m)
    if m1 + m2 + m3 != 0:
        raise ValueError(f"Wigner-3j symbols vanish unless m1 + m2 + m3 == 0; got {m}.")

    ells = np.asarray(ells, dtype=int)
    ell1 = np.asarray(ell1, dtype=int)
    ell2 = np.asarray(ell2, dtype=int)
    if np.any(ells < 0) or np.any(ell1 < 0) or np.any(ell2 < 0):
        raise ValueError("Multipoles must be non-negative.")

    # A cache hit skips the pyshtools import entirely, so a warm cache does
    # not need the optional dependency installed at all.
    path = None
    if cache_dir is not None:
        path = _w3j_cache_path(cache_dir, ells, ell1, ell2, (m1, m2, m3))
        if os.path.exists(path):
            return np.load(path, mmap_mode="r")

    shape = (ells.size, ell1.size, ell2.size)
    gib = 8.0 * shape[0] * shape[1] * shape[2] / 1024.0**3
    if gib > _W3J_WARN_GIB:
        warnings.warn(
            f"precompute_w3j is allocating a dense {shape} float64 block "
            f"({gib:.1f} GiB). Narrow the filter bands or the output ells.",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        from pyshtools.utils import Wigner3j
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise ImportError(
            "Skew spectra need Wigner-3j symbols from pyshtools, an optional "
            "dependency: `pip install bihalofit-jax[skew]`."
        ) from exc

    out = np.zeros(shape)
    n_j = int(ells.max()) + 1 if ells.size else 0
    row = np.zeros(n_j)
    for i1, L1 in enumerate(ell1):
        if abs(m1) > L1:
            continue
        for i2, L2 in enumerate(ell2):
            if abs(m2) > L2:
                continue
            w3j, jmin, jmax = Wigner3j(int(L1), int(L2), m3, m1, m2)
            row[:] = 0.0
            # w3j is indexed from jmin; place it at absolute j. (The reference
            # implementation rolls with np.concatenate instead, which corrupts
            # the row when jmin == 0 -- w3j[-0:] is the whole array.)
            if jmin < n_j:
                hi = min(int(jmax), n_j - 1)
                row[jmin : hi + 1] = w3j[: hi - jmin + 1]
            out[:, i1, i2] = row[ells]

    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        # Write then rename, so an interrupted run cannot leave a torn file
        # that a later run would read back as valid.
        tmp = f"{path}.tmp{os.getpid()}"
        # Write through a handle: np.save appends '.npy' to a *path* that
        # lacks it, which would leave the rename looking for the wrong file.
        with open(tmp, "wb") as fh:
            np.save(fh, out)
        os.replace(tmp, path)
    return out


def _build_weights(ells, band1, band2, F1, F2, m_a, m_b, cache_dir=None):
    r"""Static weight of the sum, sparsified to its support.

    ``W = (2 l1 + 1)(2 l2 + 1)/4pi * w3j_a * w3j_b`` over ``band1 x band2``,
    times ``F1 x F2`` when given, keeping the non-zero entries. The 3j symbols
    enforce the triangle inequality and parity for free, so what survives is
    exactly what contributes. ``F1``/``F2`` are ``None`` when the filters are
    traced; a top-hat is 1 throughout its support and culls nothing, so
    leaving it out does not change which triples survive.

    Parameters
    ----------
    ells : (n_ell,) int ndarray
        Output multipoles.
    band1, band2 : int ndarray
        Summed multipoles of legs 1 and 2.
    F1, F2 : ndarray or None
        Static filter values on ``band1``/``band2``, or None when traced.
    m_a, m_b : tuple of 3 int
        m values of the two 3j symbols.
    cache_dir : str or None, optional

    Returns
    -------
    L, L1, L2 : (n_nz,) int ndarray
        The surviving (ell, ell1, ell2) triples.
    w : (n_nz,) float ndarray
        Their weights.
    seg : (n_nz,) int32 ndarray
        Each triple's index into ``ells``.
    """
    w3j_a = precompute_w3j(ells, band1, band2, m_a, cache_dir=cache_dir)
    w3j_b = (
        w3j_a
        if tuple(m_b) == tuple(m_a)
        else precompute_w3j(ells, band1, band2, m_b, cache_dir=cache_dir)
    )

    prefactor = np.outer(2.0 * band1 + 1.0, 2.0 * band2 + 1.0) / (4.0 * np.pi)
    if F1 is not None:
        prefactor = prefactor * np.outer(F1, F2)
    W = w3j_a * w3j_b * prefactor[np.newaxis, :, :]

    seg, i1, i2 = np.nonzero(W)
    if seg.size == 0:
        raise ValueError(
            "The skew-spectrum weight is identically zero -- no (ell, ell1, "
            "ell2) satisfies the triangle inequality and the m selection "
            "rules. Check the filter bands are wide enough to reach the ells."
        )
    return (ells[seg], band1[i1], band2[i2], W[seg, i1, i2], seg.astype(np.int32))


def build_ell_nodes(l_min, l_max, spacing="auto", n=None):
    r"""Integer :math:`\ell` nodes for the interpolation grid.

    ``spacing`` is ``'linear'``, ``'log'``, ``'hybrid'`` (the reference
    implementation's set: integers below 10, 20 linear points to 96, then
    ``n`` log-spaced), or ``'auto'`` (default) which picks ``'hybrid'`` below
    ``l_min = 10`` and ``'linear'`` above. ``n`` is the node count, or
    ``None`` to derive it from ``l_min``.

    Accuracy is set by the **absolute** node spacing at high
    :math:`\ell_1,\ell_2`: the triangle condition confines the contributing
    configurations to a sliver of width :math:`\ell`, and a sliver narrower
    than one cell is interpolated across off-triangle nodes. So the default
    ``n`` targets a spacing of ``l_min/2``, capped at ``_N_NODES_MAX`` since
    the grid is a cube. That is also why ``'linear'`` beats ``'log'`` once
    ``l_min`` clears ~10 -- log spacing is coarsest in absolute terms exactly
    where it must be finest. Measured on a 20-1000 band at 68 nodes: linear
    5.2%, hybrid 13.4%, log 19.3%.

    Parameters
    ----------
    l_min, l_max : int
        Range to cover, ``1 <= l_min < l_max``.
    spacing : {'auto', 'linear', 'log', 'hybrid'}, optional
    n : int or None, optional

    Returns
    -------
    nodes : int ndarray
        Sorted, unique, including both ends.
    """
    l_min, l_max = int(l_min), int(l_max)
    if l_min < 1:
        raise ValueError(f"Log-spaced in ell, so needs l_min >= 1; got {l_min}.")
    if l_max <= l_min:
        raise ValueError(f"Need l_max > l_min, got {l_min}, {l_max}.")
    if spacing == "auto":
        spacing = "hybrid" if l_min < _HYBRID_BELOW else "linear"

    if spacing == "hybrid":
        n_log = 50 if n is None else int(n)
        pieces = [np.arange(l_min, min(10, l_max + 1))]
        if l_max > 10:
            pieces.append(np.linspace(10, min(96, l_max), 20))
        if l_max > 96:
            pieces.append(np.geomspace(100, l_max, n_log))
        nodes = np.concatenate(pieces)
    elif spacing in ("linear", "log"):
        if n is None:
            # spacing ~ l_min/2 at the top of the range (see the docstring).
            n = int(np.clip(2 * (l_max - l_min) // l_min + 1, 30, _N_NODES_MAX))
        grid = np.linspace if spacing == "linear" else np.geomspace
        nodes = grid(l_min, l_max, int(n))
    else:
        raise ValueError(
            f"spacing must be 'auto', 'linear', 'log' or 'hybrid'; got {spacing!r}."
        )

    nodes = np.unique(nodes.astype(int))
    nodes = nodes[(nodes >= l_min) & (nodes <= l_max)]
    return np.unique(np.concatenate([[l_min], nodes, [l_max]]))


def _node_coords(nodes, L1, L2, L):
    """Position of each query triple on the log-spaced node grid, as
    continuous indices -- the ``(3, n_nz)`` input map_coordinates wants.

    np.interp does the log-linear inversion, and clamps at the ends, so the
    coordinates are always in range.

    Parameters
    ----------
    nodes : (n_nodes,) int ndarray
        Interpolation nodes.
    L1, L2, L : (n_nz,) int ndarray
        Query triples.

    Returns
    -------
    coords : (3, n_nz) ndarray
        Continuous node indices.
    """
    log_nodes = np.log(nodes.astype(float))
    index = np.arange(nodes.size, dtype=float)
    return np.stack(
        [np.interp(np.log(q.astype(float)), log_nodes, index) for q in (L1, L2, L)]
    )


def _filter_pair(filters):
    """Per-call ``filters`` -> a pair. Same one-or-two rule as at build time.

    Takes one filter or a 2-tuple; returns a 2-tuple ``(filter1, filter2)``.
    """
    # Only a plain tuple is a per-leg pair (see _filter_spec).
    if type(filters) is not tuple:
        return filters, filters
    if len(filters) != 2:
        raise ValueError(
            f"filters must be one filter or a 2-tuple; got {len(filters)}."
        )
    return filters


def make_skew_spectra(
    ells,
    filters,
    dndz,
    tracers,
    bias=(None, None, None),
    z_source=(1089.0, 1089.0, 1089.0),
    m_a=(0, 0, 0),
    m_b=None,
    ell_sum=None,
    interp=True,
    nodes=None,
    cache_dir=None,
    pk_source="syren",
    N_chi=160,
    chi_min=_CHI_MIN_DEFAULT,
    chi_max=_CHI_MAX_DEFAULT,
    N_R=15,
    tol=None,
    R_min=1e-6,
    R_max=40.0,
    dtype=np.float64,
    term="full",
    baryonic_boost=False,
    k_lin=None,
    N_z_bg=2000,
    z_pivot=0.62,
):
    r"""Build a skew-spectrum callable for fixed multipoles, filters and tracers.

    The Wigner-3j weights, the projected-bispectrum set-up and JIT compilation happen once; call
    the result at many cosmologies.

    Parameters
    ----------
    ells : (n_ell,) int array_like
        Output multipoles, all >= 1, not necessarily contiguous.
    filters : filter, or 2-tuple of filters
        Applied to legs 1 and 2 (one filter for both, or one each); leg 3 carries the output
        multipole. A filter is anything callable on :math:`\ell`, e.g. a ``NamedTuple`` with
        ``__call__`` whose fields are its differentiable parameters; :class:`TopHat` (static) and
        :class:`SoftTopHat` (differentiable edges) are supplied. A 1-D array indexed from
        :math:`\ell=0` is accepted as static.
    dndz, tracers, bias, z_source
        As in :func:`~bihalofit_jax.make_projected_bihalofit`.
    m_a, m_b : tuple of 3 int, optional
        :math:`(m_1, m_2, m)` of the two 3j symbols, each summing to zero. ``m_b=None`` means
        ``m_a``, the scalar case.
    ell_sum : (int, int) or None, optional
        The :math:`(\ell_1, \ell_2)` range summed over, fixed at build time. ``None`` derives it
        from a :class:`TopHat` or :class:`SoftTopHat`; other filters need it. When varying a band,
        it must cover every band explored: a filter outside it is truncated.
    interp : bool, optional
        True (default) interpolates :math:`\log|B|` off a node cube; False sums :math:`B` on every
        triple, exactly. The interpolation error is largest in the squeezed limit; check a node
        set against ``interp=False``.
    nodes : (n_nodes,) int array_like or None, optional
        Interpolation nodes for all three axes; ``None`` uses :func:`build_ell_nodes`.
    cache_dir : str or None, optional
        Directory to cache the 3j blocks in (off by default: they reach several GiB).
    pk_source, N_chi, chi_min, chi_max, N_R, tol, R_min, R_max, dtype, term, baryonic_boost, k_lin, N_z_bg, z_pivot
        As in :func:`~bihalofit_jax.make_projected_bihalofit`, except that ``dtype`` defaults to
        float64: the sum has millions of terms of alternating sign.

    Returns
    -------
    skew_spectra : callable
        ``skew_spectra(cosmo, filters=None, ia=None)`` -> (N_cosmo, n_ell). ``filters`` overrides
        a non-static build-time filter; ``ia`` is an :class:`~bihalofit_jax.IA`, required exactly
        when a leg is ``'ia'``. Differentiable in the cosmology and the filter::

            g = jax.grad(lambda f: skew(cosmo, f).sum())(filt)

    Notes
    -----
    :math:`C_\ell` is linear in the filter and in :math:`B` separately. The reference
    implementation's extra :math:`(2\ell+1)` cancels for per-:math:`\ell` output; binning into
    bandpowers is left to the caller.
    """
    ells = np.unique(np.asarray(ells, dtype=int))
    if ells.size == 0:
        raise ValueError("ells is empty.")
    if ells[0] < 1:
        raise ValueError(
            f"Output multipoles must be >= 1 (ell = 0 carries no skew-spectrum "
            f"information and breaks the log-ell interpolation); got {ells[0]}."
        )

    if m_b is None:
        m_b = m_a

    (f1, band1, static1), (f2, band2, static2) = _filter_spec(filters, ell_sum)
    static_filters = static1 and static2
    if band1.size == 0 or band2.size == 0:
        raise ValueError("A filter has empty support.")

    # Static filters fold into the weight; traced ones are applied per call.
    if static_filters:
        F1, F2 = _evaluate(f1, band1), _evaluate(f2, band2)
    else:
        F1 = F2 = None

    L, L1, L2, w, seg = _build_weights(
        ells, band1, band2, F1, F2, m_a, m_b, cache_dir=cache_dir
    )

    if interp:
        if nodes is None:
            nodes = build_ell_nodes(
                min(L.min(), L1.min(), L2.min()), max(L.max(), L1.max(), L2.max())
            )
        nodes = np.unique(np.asarray(nodes, dtype=int))
        # B is evaluated on the full node outer product, ordered (l1, l2, l)
        # to match the leg convention, then flattened.
        N1, N2, N3 = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        b_ell1, b_ell2, b_ell3 = N1.ravel(), N2.ravel(), N3.ravel()
        coords = _node_coords(nodes, L1, L2, L)
    else:
        b_ell1, b_ell2, b_ell3 = L1, L2, L

    projected_bihalofit = make_projected_bihalofit(
        b_ell1,
        b_ell2,
        b_ell3,
        dndz,
        tracers,
        bias=bias,
        z_source=z_source,
        pk_source=pk_source,
        N_chi=N_chi,
        chi_min=chi_min,
        chi_max=chi_max,
        N_R=N_R,
        tol=tol,
        R_min=R_min,
        R_max=R_max,
        dtype=dtype,
        term=term,
        baryonic_boost=baryonic_boost,
        k_lin=k_lin,
        N_z_bg=N_z_bg,
        z_pivot=z_pivot,
    )

    jdtype = resolve_dtype(dtype)
    n_ell = ells.size
    w_j = jnp.asarray(w, dtype=jdtype)
    seg_j = jnp.asarray(seg)
    if interp:
        n_nodes = nodes.size
        coords_j = jnp.asarray(coords, dtype=jdtype)
    if not static_filters:
        # Where each triple's l1, l2 sit in their band, so the per-call filter
        # is evaluated once on the band (a few thousand values) and gathered,
        # rather than at every triple.
        i1_j = jnp.asarray(np.searchsorted(band1, L1).astype(np.int32))
        i2_j = jnp.asarray(np.searchsorted(band2, L2).astype(np.int32))
        band1_j = jnp.asarray(band1, dtype=jdtype)
        band2_j = jnp.asarray(band2, dtype=jdtype)

    def _interpolate(B):
        """B on the flattened node cube, (n_nodes**3,) -> B at the query triples, (n_nz,)."""
        # Trilinear in log|B| on the node cube, with the sign of the nearest node:
        # B is negative for an odd number of 'ia' legs (or a negative bias). The
        # floor keeps the logarithm finite at a node where B vanishes.
        cube = B.reshape(n_nodes, n_nodes, n_nodes)
        log_abs = jnp.log(jnp.maximum(jnp.abs(cube), jnp.finfo(jdtype).tiny))
        sign = map_coordinates(jnp.sign(cube), coords_j, order=0, mode="nearest")
        return sign * jnp.exp(
            map_coordinates(log_abs, coords_j, order=1, mode="nearest")
        )

    # Two variants, chosen at build time: static filters are folded into w_j.
    if static_filters:

        def _sum_over_l1l2(B):
            """One cosmology's bispectrum values -> its (n_ell,) skew spectrum."""
            if interp:
                B = _interpolate(B)
            return jnp.zeros(n_ell, dtype=jdtype).at[seg_j].add(w_j * B)

        _sum_all_cosmologies = jax.jit(jax.vmap(_sum_over_l1l2))
    else:

        def _sum_over_l1l2(B, F1v, F2v):
            """As above, with the filters applied per call so their edges stay
            differentiable. F1v, F2v are the filter values on band1 and band2."""
            if interp:
                B = _interpolate(B)
            weight = w_j * F1v[i1_j] * F2v[i2_j]
            return jnp.zeros(n_ell, dtype=jdtype).at[seg_j].add(weight * B)

        _sum_all_cosmologies = jax.jit(
            jax.vmap(_sum_over_l1l2, in_axes=(0, None, None))
        )

    def skew_spectra(cosmo, filters=None, ia=None):
        """Cosmology, optional filter override and IA -> skew spectra, (N_cosmo, n_ell).

        See the Returns section of :func:`make_skew_spectra`.
        """
        B = projected_bihalofit(cosmo, ia=ia)
        if static_filters:
            if filters is not None:
                raise ValueError(
                    "This callable was built with a static filter (TopHat or a "
                    "raw array), whose band is baked into the weights and "
                    "cannot be varied per call -- and whose edges have no "
                    "derivative anyway. Rebuild with SoftTopHat to vary or "
                    "differentiate the band."
                )
            return _sum_all_cosmologies(B)
        g1, g2 = (f1, f2) if filters is None else _filter_pair(filters)
        return _sum_all_cosmologies(B, _evaluate(g1, band1_j), _evaluate(g2, band2_j))

    return skew_spectra
