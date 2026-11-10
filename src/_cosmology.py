"""The Cosmology and IA containers.

Cosmology is a NamedTuple, hence a JAX pytree: gradients come back named (``dB.sigma8_cold``). The
kernel batches over its own N_cosmo axis rather than vmapping this container, so baccoemu can
evaluate every cosmology and redshift in one forward pass.
"""

from typing import NamedTuple, Union

import numpy as np
import jax.numpy as jnp

ArrayLike = Union[float, np.ndarray, jnp.ndarray]


class Cosmology(NamedTuple):
    """Cosmological parameters for the BiHaloFit kernel.

    Each field is a scalar or a shape ``(N_cosmo,)`` array; scalars broadcast
    against the longest field, so mixing the two is fine when scanning one
    parameter::

        Cosmology(omega_cold=np.linspace(0.25, 0.35, 50), omega_baryon=0.0492,
                  sigma8_cold=0.831, ns=0.9645, hubble=0.6727)

    Attributes
    ----------
    omega_cold : Omega_cdm + Omega_b, dimensionless (already includes baryons).
    omega_baryon : Omega_b, dimensionless.
    sigma8_cold : sigma_8 of the cold (cdm+baryon) component.
    ns : Scalar spectral index.
    hubble : h = H_0 / (100 km/s/Mpc). Required by every P(k) backend.
    neutrino_mass : Sum m_nu in eV (default 0; ignored by the 'eh' backend).
    w0, wa : Dark-energy equation of state (default w0=-1, wa=0, i.e. LCDM).
    """

    omega_cold: ArrayLike
    omega_baryon: ArrayLike
    sigma8_cold: ArrayLike
    ns: ArrayLike
    hubble: ArrayLike
    neutrino_mass: ArrayLike = 0.0
    w0: ArrayLike = -1.0
    wa: ArrayLike = 0.0

    @property
    def n_cosmo(self):
        """Number of cosmologies this container describes.

        Returns
        -------
        n : int
            Length of the longest field (1 if every field is a scalar).
        """
        return max(len(v) if np.ndim(v) else 1 for v in self)

    def as_arrays(self, jdtype):
        """Return a Cosmology whose every field is a ``(N_cosmo,)`` JAX array.

        Scalars and length-1 fields are broadcast to the common length, so the
        kernel can treat all eight uniformly.

        Parameters
        ----------
        jdtype : JAX dtype
            dtype of the returned arrays.

        Returns
        -------
        cosmo : Cosmology
            Every field a ``(N_cosmo,)`` array; a ``None`` field takes its default.
        """
        # An explicit None means "use this field's default". A required
        # field has no default, so None there is an error.
        raw = []
        for v, name in zip(self, self._fields):
            if v is None:
                if name not in self._field_defaults:
                    raise TypeError(f"Cosmology.{name} is required and cannot be None.")
                v = self._field_defaults[name]
            raw.append(v)

        vals = [jnp.atleast_1d(jnp.asarray(v, dtype=jdtype)) for v in raw]
        n = max(v.shape[0] for v in vals)
        for v, name in zip(vals, self._fields):
            if v.shape[0] not in (1, n):
                raise ValueError(
                    f"Cosmology.{name} has length {v.shape[0]}, which is "
                    f"neither 1 nor the batch size {n} implied by the other "
                    "fields; every field must be a scalar or share one length."
                )
        return Cosmology(*(jnp.broadcast_to(v, (n,)) for v in vals))


class IA(NamedTuple):
    r"""Intrinsic-alignment parameters of the nonlinear alignment (NLA) model.

    The alignment amplitude that multiplies the matter field is

    .. math::
        f_\mathrm{IA}(z) = -A_\mathrm{IA}\, C_1\rho_\mathrm{crit}\,
            \frac{\Omega_m}{D(z)}
            \left[\frac{1+z}{1+z_\mathrm{pivot}}\right]^{\eta},
            \qquad C_1\rho_\mathrm{crit} = 0.0134,

    with :math:`D(0)=1` and :math:`\Omega_m` the matter density today. It
    enters through the ``'ia'`` tracer; see :func:`~bihalofit_jax.make_projected_bihalofit`.
    Each field is a scalar or a shape ``(N_cosmo,)`` array, so a batch can
    carry a different amplitude per cosmology, and because this is a pytree
    the gradients come back named::

        jax.jacfwd(bk, argnums=3)(cosmo, None, None, IA(1.0)).A_IA

    Attributes
    ----------
    A_IA : amplitude at the pivot redshift (``A_IA > 0`` gives ``f_IA < 0``).
    eta : power-law index of the redshift evolution; ``0`` is plain NLA.
    """

    A_IA: ArrayLike
    eta: ArrayLike = 0.0


def as_ia(ia):
    """Coerce an IA, a dict of its fields, a ``(A_IA,)``/``(A_IA, eta)`` tuple or a
    bare amplitude to an IA.

    Parameters
    ----------
    ia : IA, dict, tuple/list of length 1-2, number or array of ndim <= 1

    Returns
    -------
    ia : IA
    """
    if isinstance(ia, IA):
        return ia
    if isinstance(ia, dict):
        unknown = set(ia) - set(IA._fields)
        if unknown:
            raise TypeError(
                f"unknown IA parameter(s) {sorted(unknown)}; "
                f"expected any of {list(IA._fields)}"
            )
        return IA(**ia)
    if isinstance(ia, (tuple, list)) and 1 <= len(ia) <= 2:
        return IA(*ia)
    if isinstance(ia, (int, float)) or (
        hasattr(ia, "shape") and hasattr(ia, "dtype") and ia.ndim <= 1
    ):
        return IA(ia)  # a bare amplitude: a number, an array or a JAX tracer
    raise TypeError(
        f"expected an IA (or a dict of its fields, or an (A_IA, eta) tuple), got "
        f"{type(ia).__name__}. Build one with, e.g., IA(A_IA=1.0, eta=0.0)."
    )


def as_cosmology(cosmo):
    """Coerce a Cosmology or a plain mapping of its fields to a Cosmology.

    Accepting a dict keeps ``**params``-style call sites working and gives a
    clearer error than a stray ``TypeError`` deep inside the kernel.

    Parameters
    ----------
    cosmo : Cosmology or dict
        A dict's keys must be ``Cosmology`` field names.

    Returns
    -------
    cosmo : Cosmology
    """
    if isinstance(cosmo, Cosmology):
        return cosmo
    if isinstance(cosmo, dict):
        unknown = set(cosmo) - set(Cosmology._fields)
        if unknown:
            raise TypeError(
                f"unknown cosmological parameter(s) {sorted(unknown)}; "
                f"expected any of {list(Cosmology._fields)}"
            )
        return Cosmology(**cosmo)
    raise TypeError(
        f"expected a Cosmology (or a dict of its fields), got "
        f"{type(cosmo).__name__}. Build one with, e.g., "
        "Cosmology(omega_cold=0.3156, omega_baryon=0.0492, "
        "sigma8_cold=0.831, ns=0.9645, hubble=0.6727)."
    )
