#!/usr/bin/env python
"""
Generate the golden reference file for test_projected_bihalofit.py.

Usage:
    python tests/generate_bkappa_golden.py                     # eh + syren
    python tests/generate_bkappa_golden.py --backends eh syren baccoemu

Saves tests/reference/bkappa_golden.npz.  Commit the result so CI has a
stable regression target; re-run whenever projected_bihalofit.py changes.
"""
import os
import sys
import argparse
import numpy as np

from bihalofit_jax import smail_nz, make_projected_bihalofit, Cosmology

_LENSING3 = ("lensing", "lensing", "lensing")

# Planck 2015 fiducial cosmology (dimensionless Ω, not ω=Ωh²)
COSMO = dict(
    omega_cold=0.3156,
    omega_baryon=0.0492,
    sigma8_cold=0.831,
    ns=0.9645,
    hubble=0.6727,
    neutrino_mass=0.0,
    w0=-1.0,
    wa=0.0,
)

# Same ell triangles used in test_projected_bihalofit.py
_ELL_TRIS = np.array(
    [
        [100.0, 100.0, 100.0],
        [500.0, 500.0, 500.0],
        [2000.0, 2000.0, 2000.0],
        [500.0, 500.0, 100.0],
        [1000.0, 1000.0, 200.0],
        [300.0, 500.0, 700.0],
        [200.0, 500.0, 600.0],
    ]
)
ELL1 = _ELL_TRIS[:, 0]
ELL2 = _ELL_TRIS[:, 1]
ELL3 = _ELL_TRIS[:, 2]

Z_GRID = np.linspace(0.0, 3.5, 500)
NZ_GRID = smail_nz(Z_GRID)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["eh", "syren"],
        choices=["eh", "syren", "baccoemu"],
        help="Backends to include in the golden file (default: eh syren)",
    )
    args = parser.parse_args()

    out = dict(ell1=ELL1, ell2=ELL2, ell3=ELL3, z_grid=Z_GRID, nz_grid=NZ_GRID)
    for k, v in COSMO.items():
        out[f"cosmo_{k}"] = np.float64(v)

    nz = (Z_GRID, NZ_GRID)
    for backend in args.backends:
        print(f"Computing B_κ with backend='{backend}' ...", flush=True)
        bk = make_projected_bihalofit(
            ELL1, ELL2, ELL3, (nz, nz, nz), _LENSING3, pk_source=backend, N_chi=20
        )
        B = np.asarray(bk(Cosmology(**COSMO)))
        out[f"Bkappa_ref_{backend}"] = B
        print(f"  B_κ: [{B.min():.4e}, {B.max():.4e}]  shape={B.shape}")

    outdir = os.path.join(os.path.dirname(__file__), "reference")
    outpath = os.path.join(outdir, "bkappa_golden.npz")
    os.makedirs(outdir, exist_ok=True)
    np.savez(outpath, **out)
    print(f"\nSaved → {outpath}")


if __name__ == "__main__":
    main()
