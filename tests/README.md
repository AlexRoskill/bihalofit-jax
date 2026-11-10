# Tests

```bash
pytest tests -q                         # everything (~15 CPU minutes)
pytest tests -q -m "not slow" --backend eh --backend syren   # what PR CI runs
```

Pushes to `main` run everything except the `highmem` Map3 grid test (34 GB), which runs only locally.
`test_ccl_kernel_crosscheck.py` is skipped without `pyccl`; run it when the lensing kernel or background changes.

| File | Covers |
|---|---|
| `test_regression.py` | the 3D bispectrum, backends, dtypes, Jacobians |
| `test_projected_bihalofit.py` | Limber projection, tracers, `B_kappa` |
| `test_skew_spectra.py` | skew spectra, filters, Wigner-3j cache |
| `test_aperture_statistics.py` | `<Map^3>` and its grids |
| `test_pk_backends_eh99.py` | the Eisenstein & Hu 1999 backend |
| `test_intrinsic_alignments.py` | the NLA tracer |
| `test_sigma8_of_z.py` | the sigma8(z) passed to the formula |
| `test_ccl_kernel_crosscheck.py` | the lensing kernel against CCL |

## Stored data

`reference/` holds two kinds of file:

| File | Written by | Kind |
|---|---|---|
| `bihalofit_fiducial_golden.npz` | `generate_golden.py` | our output, frozen: catches changes |
| `bkappa_golden.npz` | `generate_bkappa_golden.py` | our output, frozen: catches changes |
| `bihalofit_takahashi_reference.npz` | Takahashi's C code | external reference: tests correctness |
| `map3_threepoint_reference.npz` | `threepoint_py` | external reference: tests correctness |

Regenerate a golden file only after an intended change, checking the printed ratios first.
