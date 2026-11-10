"""
Shared fixtures and marks for the bihalofit_jax pytest suite.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=4")

import sys
import itertools
import numpy as np
import jax
import pytest

# The library never enables x64 itself; the suite does, before any JAX computation.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Available devices: discovered once at collection time
# ---------------------------------------------------------------------------

_CPU_DEVICE = jax.devices("cpu")[0]
_GPU_DEVICES = (
    jax.devices("gpu") if "gpu" in [d.platform for d in jax.devices()] else []
)
_ALL_DEVICES = [_CPU_DEVICE] + list(_GPU_DEVICES)


# ---------------------------------------------------------------------------
# Precision: pinned to float64 for the whole suite
# ---------------------------------------------------------------------------
# The suite checks float64-level properties (permutation symmetry at rtol=1e-14, the float64 golden
# files), so every builder defaults to float64 here; tests about float32 pass it explicitly. This runs
# before the test modules import the builders, hence conftest rather than a fixture.

import functools
import bihalofit_jax as _bhf


def _pin_float64(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs.setdefault("dtype", np.float64)
        return fn(*args, **kwargs)

    return wrapper


for _name in ("make_bihalofit", "make_projected_bihalofit", "make_map3"):
    _orig = getattr(_bhf, _name)
    _wrapped = _pin_float64(_orig)
    setattr(_bhf, _name, _wrapped)
    setattr(sys.modules[_orig.__module__], _name, _wrapped)


# ---------------------------------------------------------------------------
# GPU mark registration and auto-skip
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="append",
        default=None,
        choices=["eh", "syren", "baccoemu"],
        help="Limit tests to this P(k) backend; repeatable (default: all three)",
    )
    parser.addoption(
        "--device",
        default=None,
        choices=["cpu", "gpu"],
        help="Limit tests to a single device (default: all available)",
    )


def pytest_generate_tests(metafunc):
    """Parametrize backend and jax_device fixtures from CLI options."""
    if "backend" in metafunc.fixturenames:
        backends = metafunc.config.getoption("--backend") or ["eh", "syren", "baccoemu"]
        metafunc.parametrize("backend", backends)

    if "jax_device" in metafunc.fixturenames:
        selected = metafunc.config.getoption("--device", default=None)
        if selected == "cpu":
            devices = [_CPU_DEVICE]
        elif selected == "gpu":
            devices = list(_GPU_DEVICES)
        else:
            devices = _ALL_DEVICES
        metafunc.parametrize("jax_device", devices, ids=lambda d: d.platform)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: tests that require a JAX GPU device")
    config.addinivalue_line(
        "markers",
        "slow: convergence sweeps and all-parameter gradient checks; CI skips these on PRs",
    )
    config.addinivalue_line(
        "markers",
        "highmem: needs >30 GB RAM (fine Map3 reference grid); CI never runs these",
    )


# ---------------------------------------------------------------------------
# Memory cap
# ---------------------------------------------------------------------------
# Every builder compiles new XLA programs, and JAX's caches keep them alive, so
# resident memory only grows: ~13 GB by the end of the suite, which gets a 7 GB
# CI runner killed. Clearing after every test holds it under ~1.5 GB but
# doubles the run time (recompilation), so clear only past a threshold.

_MAX_RSS_MB = float(os.environ.get("BIHALOFIT_TEST_MAX_RSS_MB", 3000))


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024
    except OSError:  # not Linux
        return 0.0


def pytest_runtest_teardown(item, nextitem):
    if _rss_mb() > _MAX_RSS_MB:
        import gc, ctypes

        jax.clear_caches()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # return freed heap to the OS
        except OSError:
            pass


@pytest.fixture(autouse=True)
def skip_no_gpu(request):
    if request.node.get_closest_marker("gpu"):
        import jax

        if not jax.devices("gpu"):
            pytest.skip("no JAX GPU device available")


# ---------------------------------------------------------------------------
# Cosmology parameters (Planck 2015 fiducial)
# ---------------------------------------------------------------------------

OMEGA_B = 0.0492
OMEGA_CDM = 0.2664
OMEGA_M = OMEGA_B + OMEGA_CDM
H = 0.6727
NS = 0.9645
SIGMA8 = 0.831
W0 = -1.0
WA = 0.0


# ---------------------------------------------------------------------------
# Triangle grid helper
# ---------------------------------------------------------------------------


def _make_triangles(kmin=0.01, kmax=1.0, dlogk=0.1):
    n_bins = int(np.floor((np.log(kmax) - np.log(kmin)) / dlogk))
    edges = np.exp(np.linspace(np.log(kmin), np.log(kmin) + n_bins * dlogk, n_bins + 1))
    kc = (edges[:-1] + edges[1:]) / 2.0
    combos = np.array(list(itertools.combinations_with_replacement(kc, 3)))
    # Strict triangle inequality: exclude near-degenerate
    valid = combos[:, 2] < combos[:, 0] + combos[:, 1] * (1.0 - 1e-4)
    tris = combos[valid]
    order = np.lexsort((tris[:, 2], tris[:, 1], tris[:, 0]))
    tris = tris[order]
    return tris[:, 0].copy(), tris[:, 1].copy(), tris[:, 2].copy()


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fiducial_params():
    """Planck 2015 fiducial cosmology, N_cosmo=1."""
    return dict(
        omega_cold=np.array([OMEGA_M]),
        omega_baryon=np.array([OMEGA_B]),
        sigma8_cold=np.array([SIGMA8]),
        ns=np.array([NS]),
        hubble=np.array([H]),
        neutrino_mass=np.array([0.0]),
        w0=np.array([W0]),
        wa=np.array([WA]),
    )


@pytest.fixture(scope="session")
def golden_cosmo_params():
    """~10 cosmologies covering the baccoemu prior: used for golden generation and test 1."""
    # Each row is: [omega_cold, omega_b, sigma8, h, ns, mnu, w0, wa]
    specs = [
        # fiducial
        [OMEGA_M, OMEGA_B, SIGMA8, H, NS, 0.00, -1.0, 0.0],
        # sigma8 edges
        [OMEGA_M, OMEGA_B, 0.88, H, NS, 0.00, -1.0, 0.0],
        [OMEGA_M, OMEGA_B, 0.75, H, NS, 0.00, -1.0, 0.0],
        # omega_m edges
        [0.38, OMEGA_B, SIGMA8, H, NS, 0.00, -1.0, 0.0],
        [0.26, OMEGA_B, SIGMA8, H, NS, 0.00, -1.0, 0.0],
        # hubble edges
        [OMEGA_M, OMEGA_B, SIGMA8, 0.74, NS, 0.00, -1.0, 0.0],
        [OMEGA_M, OMEGA_B, SIGMA8, 0.62, NS, 0.00, -1.0, 0.0],
        # non-zero neutrino mass
        [OMEGA_M, OMEGA_B, SIGMA8, H, NS, 0.15, -1.0, 0.0],
        # w0 variation (within baccoemu prior ≈ [-1.15, -0.85])
        [OMEGA_M, OMEGA_B, SIGMA8, H, NS, 0.00, -0.90, 0.0],
        # wa variation
        [OMEGA_M, OMEGA_B, SIGMA8, H, NS, 0.00, -1.0, 0.25],
    ]
    oc, ob, s8, h, ns_, mnu, w0, wa = np.array(specs).T
    return dict(
        omega_cold=oc,
        omega_baryon=ob,
        sigma8_cold=s8,
        hubble=h,
        ns=ns_,
        neutrino_mass=mnu,
        w0=w0,
        wa=wa,
    )


@pytest.fixture(scope="session")
def triangles():
    """Full dlogk=0.1 triangle grid, k in [0.01, 1.0] h/Mpc (~3400 triangles)."""
    return _make_triangles()


# 9 explicit triangles covering all shape regimes: used for tests that only
# need to verify a property (dtype, permutation, Jacobian consistency) rather
# than survey a grid.  Tiny N_tri keeps XLA compilation fast regardless of backend.
_TEST_TRI = np.array(
    [
        [0.02, 0.02, 0.02],  # equilateral, linear
        [0.10, 0.10, 0.10],  # equilateral, transition
        [3.00, 3.00, 3.00],  # equilateral, nonlinear
        [0.10, 0.10, 0.02],  # squeezed
        [3.00, 3.00, 0.05],  # squeezed, high-k
        [0.10, 0.10, 0.195],  # near-degenerate
        [0.30, 0.20, 0.48],  # scalene
        [0.05, 0.15, 0.18],  # scalene, mid-k
        [0.01, 0.50, 0.50],  # extreme dynamic range
    ]
)


@pytest.fixture(scope="session")
def test_triangles():
    """9 representative triangles covering all shape regimes; keeps JIT compilation fast."""
    return _TEST_TRI[:, 0], _TEST_TRI[:, 1], _TEST_TRI[:, 2]


@pytest.fixture(scope="session")
def redshifts():
    return np.array([0.0, 0.5, 1.0, 2.0, 3.0])


_GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "reference", "bihalofit_fiducial_golden.npz"
)

_TAKAHASHI_PATH = os.path.join(
    os.path.dirname(__file__), "reference", "bihalofit_takahashi_reference.npz"
)

_MAP3_THREEPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "reference", "map3_threepoint_reference.npz"
)


@pytest.fixture(scope="session")
def golden():
    """Load the pre-generated golden reference file; skip if it doesn't exist."""
    if not os.path.exists(_GOLDEN_PATH):
        pytest.skip(
            f"golden reference file not found: {_GOLDEN_PATH}; "
            "run tests/generate_golden.py first"
        )
    return np.load(_GOLDEN_PATH, allow_pickle=True)


@pytest.fixture(scope="session")
def takahashi_reference():
    """Load the Takahashi C code reference file; skip if it doesn't exist."""
    if not os.path.exists(_TAKAHASHI_PATH):
        pytest.skip(f"Takahashi reference file not found: {_TAKAHASHI_PATH}")
    return np.load(_TAKAHASHI_PATH, allow_pickle=True)


@pytest.fixture(scope="session")
def map3_threepoint_reference():
    """Load the threepoint_py <Map_ap^3> reference file; skip if it doesn't exist."""
    if not os.path.exists(_MAP3_THREEPOINT_PATH):
        pytest.skip(
            f"threepoint_py Map3 reference file not found: {_MAP3_THREEPOINT_PATH}"
        )
    return np.load(_MAP3_THREEPOINT_PATH, allow_pickle=True)


@pytest.fixture
def jax_device(request):
    """Parametrized fixture: runs once per available device (cpu, gpu, ...)."""
    return request.param


@pytest.fixture(scope="session")
def cpu_device():
    return _CPU_DEVICE


@pytest.fixture(scope="session")
def gpu_device():
    if not _GPU_DEVICES:
        pytest.skip("no JAX GPU device available")
    return _GPU_DEVICES[0]
