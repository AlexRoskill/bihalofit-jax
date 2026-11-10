Internal API
============

This page is for reading and changing the code. Nothing here is imported from the
package root, and any of it may change without notice; to use the package, see
:doc:`api`. Each module page lists every function, private ones included.

Overview
--------

Following :func:`~bihalofit_jax.make_projected_bihalofit`, which the other projected
builders call in turn:

1. **Checks.** Options are validated (:mod:`~bihalofit_jax._validation`), and one-off
   warnings flag a :math:`\chi` range or P(k) grid that is too small for the request
   (:mod:`~bihalofit_jax._diagnostics`).
2. **Projection set-up.** :math:`\chi(z)` and the Gauss-Legendre :math:`\chi`-nodes
   (:mod:`~bihalofit_jax.background`), one radial kernel :math:`W_i(\chi)` per leg
   (:mod:`~bihalofit_jax.tracers`), and the triangles at each node,
   :math:`k_i = (\ell_i + 1/2)/\chi`.
3. **Compilation.** The bispectrum kernel is vmapped over the :math:`\chi`-nodes and
   over cosmologies, then JIT-compiled.
4. **Each call with a cosmology** runs the kernel
   :func:`~bihalofit_jax.bispectrum.bispectrum_all_z`: the linear P(k)
   (:mod:`~bihalofit_jax._pk_backends`), the nonlinear scale :math:`k_\mathrm{NL}` and
   slope :math:`n_\mathrm{eff}` (:mod:`~bihalofit_jax._knl_solver`), then the
   BiHalofit terms. The Limber sum over nodes gives :math:`B(\ell_1, \ell_2, \ell_3)`.

:func:`~bihalofit_jax.make_bihalofit` is the same without step 2: it evaluates the
kernel on fixed 3D triangles. :func:`~bihalofit_jax.make_map3` and
:func:`~bihalofit_jax.make_skew_spectra` build a projected bispectrum on their own
multipole grid and integrate it.

To read the code, start with :mod:`~bihalofit_jax.bispectrum`, then
:mod:`~bihalofit_jax.projected_bispectrum`.

Core modules
------------

Each module holds one public builder and the code specific to it.

.. autosummary::
   :toctree: generated/internal
   :template: internal_module.rst

   bihalofit_jax.bispectrum
   bihalofit_jax.projected_bispectrum
   bihalofit_jax.aperture
   bihalofit_jax.skew_spectra

Supporting modules
------------------

The ingredients of the bispectrum and of its projection.

.. autosummary::
   :toctree: generated/internal
   :template: internal_module.rst

   bihalofit_jax._pk_backends
   bihalofit_jax._knl_solver
   bihalofit_jax.background
   bihalofit_jax.tracers

Extras
------

Parameter containers, precision, input checks, build-time warnings and numerical
helpers.

.. autosummary::
   :toctree: generated/internal
   :template: internal_module.rst

   bihalofit_jax._cosmology
   bihalofit_jax._config
   bihalofit_jax._validation
   bihalofit_jax._diagnostics
   bihalofit_jax._utils
