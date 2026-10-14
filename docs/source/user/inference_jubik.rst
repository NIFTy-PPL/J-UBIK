Inference with J-UBIK and NIFTy
===============================

To do inference with different instrument, we need to load instrumental and observational data. The following demo shows how to do this for Chandra:

.. toctree::
        :maxdepth: 1

        chandra_likelihood_demo

This knowledge about the instrument and the measurement process encodes the likelihood. Combining the likelihood with the prior models described earlier, we can be build inference scripts using **NIFTy.re**. 


In the following demo we show how all the parts of J-UBIK work together: we get the observational and instrumental data, construct the instrument response function, define a prior model, define the likelihood and run inference to recover posterior samples modelling a Chandra observation of supernova remnant G299.2-2.9. 

.. toctree::
        :maxdepth: 1

        x-ray-imaging

Minimization configuration
--------------------------

``MinimizationParser`` returns iteration callables for the solver settings,
sample count and mode, constants, and point estimates. Settings may be scalars
or lists selected by their section's ``switches``. A shorter value list repeats
its last entry. Switches must start at zero and increase; the existing indexing
uses ``iteration + 1``, so a switch at 10 applies when the callable receives 9.

Explicit numeric values, including zero, override defaults. Missing values and
YAML ``null`` inherit: ``lin_absdelta``, ``nonlin_xtol``, and ``kl_absdelta`` use
``delta`` when provided; otherwise the parser omits unset solver arguments and
NIFTy supplies its defaults. ``delta`` requires ``n_dof`` and derives linear
``absdelta = delta * n_dof / 10``, nonlinear ``xtol = delta``, and KL
``absdelta = delta * n_dof``. Its switches are independent of the solver sections.

Available numeric controls are:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Section
     - Prefix
     - Suffixes
   * - ``samples``
     - ``lin_``
     - ``absdelta``, ``tol``, ``atol``, ``miniter``, ``maxiter``, ``resnorm``, ``norm_ord``
   * - ``samples``
     - ``nonlin_``
     - ``absdelta``, ``xtol``, ``miniter``, ``maxiter``, ``norm_ord``, ``energy_reduction_factor``
   * - ``samples``
     - ``nonlin_cg_``
     - ``absdelta``, ``tol``, ``atol``, ``miniter``, ``maxiter``, ``resnorm``, ``norm_ord``
   * - ``kl_minimization``
     - ``kl_``
     - ``absdelta``, ``xtol``, ``miniter``, ``maxiter``, ``norm_ord``, ``energy_reduction_factor``
   * - ``kl_minimization``
     - ``kl_cg_``
     - ``absdelta``, ``tol``, ``atol``, ``miniter``, ``maxiter``, ``resnorm``, ``norm_ord``

For example, ``kl_cg_resnorm: [0.1, 0.01]`` configures two CG residual thresholds.
CG uses ``tol`` and ``atol`` only when neither ``absdelta`` nor ``resnorm`` is
active. Newton supplies an adaptive inner CG residual threshold, and can also
supply an energy threshold. An explicit ``*_cg_absdelta`` or ``*_cg_resnorm``
overrides the respective calculated threshold.

.. note::

   The parser previously forwarded unset options as ``None``, which disabled
   Newton's calculated CG energy threshold. Omitting unset options restores
   native NIFTy behavior. Existing configurations can therefore use different
   solver iterations and produce different numerical results, even with the
   same outer tolerances. ``null`` now inherits rather than disabling a criterion.

Domain keys use nested lists to distinguish key groups from schedule entries::

    constants:
      switches: [0, 10]
      domain_keys: [[diffuse_offset_mean], null]
    point_estimates:
      domain_keys: [[point_source_amplitude]]

The callables return tuples, or ``None`` when inactive. Pass them as
``constants=parser.constants`` and ``point_estimates=parser.point_estimates``
to ``nifty.re.optimize_kl``. Constants freeze parameters during KL minimization;
point estimates exclude parameters from posterior sampling.
