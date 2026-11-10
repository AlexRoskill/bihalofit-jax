# bihalofit-JAX

JAX implementation of the BiHaloFit nonlinear matter bispectrum ([Takahashi et al. 2020](https://arxiv.org/abs/1911.07886)), with fully differentiable P(k) backends and GPU support.
This package provides support for:
- The 3D matter bispectrum, $B_\delta(k_1,k_2,k_3,z)$.
- The Limber-projected angular bispectrum for a
  tomographic sample. Each triangle leg picks its own projection tracer
  (`'lensing'`, `'counts'` for galaxy clustering, `'cmb_lensing'`, or `'ia'`
  for intrinsic alignments), so the same function covers cosmic shear,
  galaxy-galaxy lensing, CMB-lensing cross bispectra and the NLA alignment
  terms. See `notebooks/tutorials/projected_bihalofit_tutorial.ipynb`.
- Third-order aperture-mass statistics.
  $\langle X_1X_2X_3\rangle(\theta_1,\theta_2,\theta_3)$ from the projected
  bispectrum. See `notebooks/tutorials/aperture_statistics_tutorial.ipynb`.
- Skew spectra. The bispectrum compressed onto a single multipole,
  cross-correlating one field with a filtered quadratic combination of two
  others; filters are differentiable JAX pytrees, so band shapes can be
  optimised. See `notebooks/tutorials/skew_spectra_tutorial.ipynb`.

## Install

```bash
pip install bihalofit-jax                 # eh, eh_99 and syren backends
pip install "bihalofit-jax[baccoemu]"     # also the baccoemu backend
pip install -e ".[dev]"                   # from a clone: editable, with the test dependencies
```

The skew spectra need `pyshtools` (`pip install "bihalofit-jax[skew]"`); `jax[cuda12]` gives GPU support.

## Quick start

```python
import numpy as np
import jax
import bihalofit_jax
from bihalofit_jax import Cosmology, make_bihalofit

# Input k vector
k = np.array([0.1, 0.2, 0.3])  # h/Mpc

# Build cosmology
cosmo = Cosmology(omega_cold=0.3156, omega_baryon=0.0492,
                  sigma8_cold=0.831, ns=0.9645, hubble=0.6727)

# Build bihalofit callable
bhf = make_bihalofit(k1=k, k2=k, k3=k, expfactors=np.array([1.0]))
# Call bihalofit for given cosmology
B = bhf(cosmo)

# Evaluating gradients
dB = jax.grad(lambda c: bhf(c).sum())(cosmo)   # Cosmology is a pytree: gradients come back named
dB.sigma8_cold
```

See `notebooks/tutorials/bihalofit_jax_tutorial.ipynb` for the full walkthrough,
including multiple cosmologies, multiple redshifts and automatic
differentiation.

## Backends

| `pk_source` | Linear P(k) | Extra dependency |
|---|---|---|
| `'syren'` (default) | Symbolic regression (wCDM + neutrinos) | none |
| `'baccoemu'` | baccoemu neural-network emulator | `baccoemu` |
| `'eh'` | Eisenstein & Hu 1998 no-wiggle fitting formula | none |
| `'eh_99'` | Eisenstein & Hu 1999 CDM+HDM fitting formula (at f&#957;=0) | none |

## Repository layout

```
src/                 the package itself
tests/               pytest suite
notebooks/           the four tutorials
docs/                the Sphinx site
```

Each of them except `src/` has its own README.

## Citation

If you use this package, please cite the BiHalofit paper
```bibtex
@article{Takahashi2020,
  author  = {Takahashi, Ryuichi and Nishimichi, Takahiro and Namikawa, Toshiya and
             Taruya, Atsushi and Kayo, Issha and Osato, Ken and Kobayashi, Yosuke and
             Shirasaki, Masato},
  title   = {Fitting the nonlinear matter bispectrum by the Halofit approach},
  journal = {The Astrophysical Journal},
  volume  = {895},
  number  = {2},
  pages   = {113},
  year    = {2020},
  doi     = {10.3847/1538-4357/ab908d},
  eprint  = {1911.07886}
}
```

and the paper for the linear P(k) backend you use:
[baccoemu (Aricò et al. 2021)](https://arxiv.org/abs/2104.14568),
[syren (Sui et al. 2025)](https://arxiv.org/abs/2410.14623),
[Eisenstein & Hu 1998](https://arxiv.org/abs/astro-ph/9709112), or
[Eisenstein & Hu 1999](https://arxiv.org/abs/astro-ph/9710252).

Note that the skew-spectrum implementation follows
[Andrina Nicola et al.'s `bispectrum`](https://github.com/anicola/bispectrum).