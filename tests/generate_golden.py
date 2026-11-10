"""
Developer script: generate the golden reference file for test_regression.py.

Run ONCE (after deliberate algorithm changes) from the repo root:
    python tests/generate_golden.py

Inspect the printed max/median ratios between backends. If they look reasonable
(baccoemu/E&H < 25%, baccoemu/syren < 5%), commit the output file:
    tests/reference/bihalofit_fiducial_golden.npz

Re-running this script overwrites the existing golden file: only do so after a
deliberate change to the algorithm (not just a refactor).
"""

import sys
import os
import itertools

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import jax

jax.config.update(
    "jax_enable_x64", True
)  # float64 where the package allows it; B is built (and stored) at its default float32
from bihalofit_jax import make_bihalofit


# ---------------------------------------------------------------------------
# Cosmologies: ~10 covering the baccoemu prior edges
# ---------------------------------------------------------------------------

OMEGA_B = 0.0492
OMEGA_CDM = 0.2664
OMEGA_M = OMEGA_B + OMEGA_CDM
H = 0.6727
NS = 0.9645
SIGMA8 = 0.831

specs = [
    # [omega_cold, omega_b, sigma8, h, ns, mnu, w0, wa]
    [OMEGA_M, OMEGA_B, SIGMA8, H, NS, 0.00, -1.0, 0.0],  # fiducial
    [OMEGA_M, OMEGA_B, 0.88, H, NS, 0.00, -1.0, 0.0],  # high s8
    [OMEGA_M, OMEGA_B, 0.75, H, NS, 0.00, -1.0, 0.0],  # low s8
    [0.38, OMEGA_B, SIGMA8, H, NS, 0.00, -1.0, 0.0],  # high om
    [0.26, OMEGA_B, SIGMA8, H, NS, 0.00, -1.0, 0.0],  # low om
    [OMEGA_M, OMEGA_B, SIGMA8, 0.74, NS, 0.00, -1.0, 0.0],  # high h
    [OMEGA_M, OMEGA_B, SIGMA8, 0.62, NS, 0.00, -1.0, 0.0],  # low h
    [OMEGA_M, OMEGA_B, SIGMA8, H, NS, 0.15, -1.0, 0.0],  # mnu
    [OMEGA_M, OMEGA_B, SIGMA8, H, NS, 0.00, -0.90, 0.0],  # w0
    [OMEGA_M, OMEGA_B, SIGMA8, H, NS, 0.00, -1.0, 0.25],  # wa
]

oc, ob, s8, h, ns_, mnu, w0, wa = np.array(specs).T
cosmo_params = dict(
    omega_cold=oc,
    omega_baryon=ob,
    sigma8_cold=s8,
    hubble=h,
    ns=ns_,
    neutrino_mass=mnu,
    w0=w0,
    wa=wa,
)
N_cosmo = len(specs)


# ---------------------------------------------------------------------------
# Triangle grid: dlogk=0.1, k in [0.01, 1.0] h/Mpc
# ---------------------------------------------------------------------------

KMIN, KMAX, DLOGK = 0.01, 1.0, 0.1
n_bins = int(np.floor((np.log(KMAX) - np.log(KMIN)) / DLOGK))
edges = np.exp(np.linspace(np.log(KMIN), np.log(KMIN) + n_bins * DLOGK, n_bins + 1))
kc = (edges[:-1] + edges[1:]) / 2.0
combos = np.array(list(itertools.combinations_with_replacement(kc, 3)))
valid = combos[:, 2] < combos[:, 0] + combos[:, 1] * (1.0 - 1e-4)
tris = combos[valid]
order = np.lexsort((tris[:, 2], tris[:, 1], tris[:, 0]))
tris = tris[order]
k1 = tris[:, 0].copy()
k2 = tris[:, 1].copy()
k3 = tris[:, 2].copy()
print(f"Triangles: {len(k1)} (dlogk={DLOGK}, k in [{KMIN}, {KMAX}] h/Mpc)")


# ---------------------------------------------------------------------------
# Redshifts
# ---------------------------------------------------------------------------

REDSHIFTS = np.array([0.0, 0.5, 1.0, 2.0])
A_VALS = 1.0 / (1.0 + REDSHIFTS)


# ---------------------------------------------------------------------------
# Evaluate all three backends
# ---------------------------------------------------------------------------


def _run(backend):
    print(f"\n  Running backend '{backend}' for {N_cosmo} cosmologies ...", flush=True)
    B = make_bihalofit(k1, k2, k3, A_VALS, pk_source=backend)(cosmo_params)
    import numpy as np

    B_np = np.asarray(B)
    assert np.all(np.isfinite(B_np)), f"Non-finite values in {backend} output"
    assert np.all(B_np > 0), f"Non-positive values in {backend} output"
    print(f"  Done. B range: [{B_np.min():.3e}, {B_np.max():.3e}] (Mpc/h)^6")
    return B_np


print("Generating golden reference file...")
B_bemu = _run("baccoemu")
B_eh = _run("eh")
B_syren = _run("syren")


# ---------------------------------------------------------------------------
# Sanity ratios: inspect before committing
# ---------------------------------------------------------------------------


def _ratio_stats(B_a, B_b, name_a, name_b):
    ratio = B_a / B_b
    print(f"\n  {name_a} / {name_b}:")
    print(f"    max  |ratio - 1| = {np.max( np.abs(ratio - 1)):.4f}")
    print(f"    median ratio     = {np.median(ratio):.6f}")


print("\n--- Backend ratio check (inspect before committing) ---")
_ratio_stats(B_bemu, B_syren, "baccoemu", "syren")
_ratio_stats(B_bemu, B_eh, "baccoemu", "E&H")
_ratio_stats(B_syren, B_eh, "syren", "E&H")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_dir = os.path.join(_HERE, "reference")
out_path = os.path.join(out_dir, "bihalofit_fiducial_golden.npz")
os.makedirs(out_dir, exist_ok=True)

np.savez(
    out_path,
    k1=k1,
    k2=k2,
    k3=k3,
    redshifts=REDSHIFTS,
    B_ref_baccoemu=B_bemu,
    B_ref_eh=B_eh,
    B_ref_syren=B_syren,
    **{f"cosmo_{k}": v for k, v in cosmo_params.items()},
)

size_mb = os.path.getsize(out_path) / 1e6
print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
print("Inspect the ratios above. If they look reasonable, commit this file.")
