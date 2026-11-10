Public API
==========

Everything on this page is imported from the package root, for example
``from bihalofit_jax import Cosmology, make_bihalofit``.

Overview
--------

The main functionalities of the package are the following:

1. **Linear power spectrum** :math:`P_\mathrm{L}(k)`, from the backend chosen with
   ``pk_source``: ``'syren'`` (default), ``'baccoemu'``, ``'eh'`` or ``'eh_99'``.
2. **3D matter bispectrum** :math:`B(k_1, k_2, k_3; z)` from the BiHalofit fitting
   formula: :func:`~bihalofit_jax.make_bihalofit`.
3. **Projected bispectrum** :math:`B(\ell_1, \ell_2, \ell_3)`, the Limber projection of
   step 2 with its own tracer on each leg (lensing, galaxy counts, CMB lensing,
   intrinsic alignments): :func:`~bihalofit_jax.make_projected_bihalofit`.
4. **Third-order aperture statistics** :math:`\langle M_\mathrm{ap}^3 \rangle`:
   :func:`~bihalofit_jax.make_map3`.
5. **Skew spectra**, the power spectra of the squared field with the original field,
   for example :math:`C_\ell^{g^2 g}`: :func:`~bihalofit_jax.make_skew_spectra`.


Cosmology
---------

The argument of every returned function. Pass arrays of length :math:`N_\mathrm{cosmo}`
to evaluate many cosmologies in one call. :class:`~bihalofit_jax.IA` is needed only
for ``'ia'`` tracers.

.. autosummary::
   :toctree: generated
   :nosignatures:

   bihalofit_jax.Cosmology
   bihalofit_jax.IA

Summary Statistics
------------------

The summary statistics are the main outputs of the package. They are designed to 
be constructed once and called many times with different cosmologies. The projected builders
also take a source redshift distribution and optional tracer parameters.

.. autosummary::
   :toctree: generated
   :nosignatures:

   bihalofit_jax.make_bihalofit
   bihalofit_jax.make_projected_bihalofit
   bihalofit_jax.make_map3
   bihalofit_jax.make_skew_spectra

Inputs to statistics
--------------------

Helpers for the arguments the projected builders need:

* a source redshift distribution :math:`n(z)`: :func:`~bihalofit_jax.smail_nz`;
* aperture radii for :func:`~bihalofit_jax.make_map3`, which takes radians:
  multiply arcminutes by :data:`~bihalofit_jax.ARCMIN_TO_RAD`;
* band filters and multipole nodes for :func:`~bihalofit_jax.make_skew_spectra`:
  :class:`~bihalofit_jax.TopHat`, :class:`~bihalofit_jax.SoftTopHat` and
  :func:`~bihalofit_jax.build_ell_nodes`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   bihalofit_jax.smail_nz
   bihalofit_jax.ARCMIN_TO_RAD
   bihalofit_jax.TopHat
   bihalofit_jax.SoftTopHat
   bihalofit_jax.build_ell_nodes

Precision
---------

The builders compute in float32 by default (``make_skew_spectra`` in float64). For
float64, call :func:`~bihalofit_jax.enable_x64` once at the start of the script and
pass ``dtype=np.float64``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   bihalofit_jax.enable_x64

The code behind these functions, private functions included, is described in
:doc:`api_internal`.
