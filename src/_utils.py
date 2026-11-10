"""Small numerical helpers: Gauss-Legendre rules, coercion, C1 (Catmull-Rom) interpolation and a smooth minimum, which keep
kinked quantities smooth for HMC without changing their values beyond O(dx^2) and eps/2.
"""

import numpy as np
import jax.numpy as jnp


def gauss_legendre(a, b, N):
    """Gauss-Legendre nodes and weights on [a, b]: ``sum(w * f(x)) ~= int_a^b f dx``.

    Parameters
    ----------
    a, b : float
        Integration limits.
    N : int
        Number of nodes.

    Returns
    -------
    x, w : (N,) ndarray
        Nodes and weights.
    """
    xi, wi = np.polynomial.legendre.leggauss(N)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    return mid + half * xi, wi * half


def gauss_legendre_log(a, b, N):
    """Gauss-Legendre nodes log-uniform on [a, b], with weights for ``int_a^b f dx``.

    Parameters
    ----------
    a, b : float
        Integration limits, both positive.
    N : int
        Number of nodes.

    Returns
    -------
    x, w : (N,) ndarray
        Nodes and weights (the Jacobian ``x`` is folded into ``w``).
    """
    xi, wi = np.polynomial.legendre.leggauss(N)
    lo, hi = np.log(a), np.log(b)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    x = np.exp(mid + half * xi)
    return x, wi * half * x


def _as_jax_vector(v, jdtype, default=None):
    """Coerce a scalar/array (or its default) to a JAX (N,) vector.

    Parameters
    ----------
    v : scalar, array-like or None
        Value to coerce; None uses ``default``.
    jdtype : JAX dtype
    default : scalar or array-like, optional

    Returns
    -------
    vec : (N,) jax array
        At least 1-D.
    """
    if v is None:
        v = default
    return jnp.atleast_1d(jnp.asarray(v, dtype=jdtype))


def _crom_eval(Y, xi):
    """Catmull-Rom (C¹) interpolation of tabulated values at a continuous grid index.

    Parameters
    ----------
    Y : (N,) or (N, N_cosmo)
        Values on a grid, indexed along axis 0 (one column per cosmology if 2-D).
    xi : array, any shape if Y is 1-D; (N_cosmo,) if Y is 2-D
        Continuous index into the grid (xi = 2.5 is halfway between Y[2] and Y[3]);
        valid on [1, N-2], the cubic needs one neighbour either side.

    Returns
    -------
    val : shape of xi
        Interpolated Y at xi.
    der : shape of xi
        dY/dxi at xi (per unit index, not per unit of the grid coordinate).
    """
    N = Y.shape[0]
    i = jnp.clip(jnp.floor(xi).astype(jnp.int32), 1, N - 3)
    u = xi - i
    if Y.ndim == 1:
        y0, y1, y2, y3 = Y[i - 1], Y[i], Y[i + 1], Y[i + 2]
    else:
        cc = jnp.arange(Y.shape[1])
        y0, y1, y2, y3 = Y[i - 1, cc], Y[i, cc], Y[i + 1, cc], Y[i + 2, cc]
    a0 = y1
    a1 = 0.5 * (-y0 + y2)
    a2 = 0.5 * (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3)
    a3 = 0.5 * (-y0 + 3.0 * y1 - 3.0 * y2 + y3)
    val = a0 + u * (a1 + u * (a2 + u * a3))
    der = a1 + u * (2.0 * a2 + 3.0 * u * a3)
    return val, der


def _c1_newton_invert(Y, target, xi_seed, N, n_steps=8):
    """Continuous index xi at which the Catmull-Rom interpolant of Y equals target.

    Newton on _crom_eval, so xi is differentiable in Y and target (a
    searchsorted bracket would jump as they move).

    Parameters
    ----------
    Y : (N,)
        Monotonically increasing grid values, e.g. chi(z) on the background z-grid.
    target : array
        Values to locate in Y, e.g. the chi-nodes.
    xi_seed : shape of target
        Starting index, e.g. from linear interpolation in the bracketing interval.
    N : int
        len(Y), static.
    n_steps : int
        Fixed number of Newton steps (no convergence check).

    Returns
    -------
    xi : shape of target
        Continuous index, clipped to [1, N-2].
    """
    xi = xi_seed
    for _ in range(n_steps):
        y_xi, dy_xi = _crom_eval(Y, xi)
        xi = xi - (y_xi - target) / dy_xi
    return jnp.clip(xi, 1.0, N - 2.0)


def _smooth_min(a, b, eps):
    """C-infinity approximation to min(a, b); max deviation eps/2 at a==b.

    Parameters
    ----------
    a, b : array
        Broadcastable arguments.
    eps : float
        Smoothing width.

    Returns
    -------
    m : broadcast shape of a and b
    """
    return 0.5 * (a + b - jnp.sqrt((a - b) ** 2 + eps**2))
