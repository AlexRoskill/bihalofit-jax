"""bihalofit-JAX: the BiHalofit matter bispectrum (Takahashi et al. 2020, arXiv:1911.07886) in JAX,
and its projected statistics.

Each layer is a ``make_*`` builder: it does the set-up and compilation once for a fixed geometry and
returns a callable of a :class:`Cosmology`, a JAX pytree, so gradients come back named::

    model = make_bihalofit(k1, k2, k3, expfactors)
    B = model(cosmo)
    dB = jax.grad(lambda c: model(c).sum())(cosmo)      # dB.sigma8_cold, ...

  make_bihalofit            B(k1, k2, k3) in (Mpc/h)^6, k in h/Mpc
  make_projected_bihalofit  Limber-projected B(l1, l2, l3), one tracer per leg
  make_map3                 third-order aperture statistics
  make_skew_spectra         skew spectra C_l^{(jk)i}

The builders default to float32 (``make_skew_spectra`` to float64). For float64 call
:func:`enable_x64` once, or set ``JAX_ENABLE_X64=1``; the package never changes global JAX state.
Underscore-prefixed modules are internal.
"""

__version__ = "0.1.0"  # read by pyproject.toml; keep CITATION.cff in step

from ._config import enable_x64
from ._cosmology import Cosmology, IA
from .bispectrum import make_bihalofit
from .projected_bispectrum import make_projected_bihalofit, smail_nz
from .aperture import make_map3, ARCMIN_TO_RAD
from .skew_spectra import make_skew_spectra, TopHat, SoftTopHat, build_ell_nodes

__all__ = [
    "__version__",
    "enable_x64",
    "Cosmology",
    "IA",
    # one builder per layer: build once, call with many cosmologies
    "make_bihalofit",
    "make_projected_bihalofit",
    "make_map3",
    "make_skew_spectra",
    # inputs those builders require
    "smail_nz",
    "ARCMIN_TO_RAD",
    "TopHat",
    "SoftTopHat",
    "build_ell_nodes",
]
