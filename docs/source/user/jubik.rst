What is J-UBIK?
===============

**J-UBIK** is built on top of **NIFTy.re**, a software package designed for the accelerated Bayesian inference of physical fields. Currently, the scope of application is limited to a few astrophysical instruments, but there are plans to expand the areas of application further. The main role of **J-UBIK** is to model the prior in a generative fashion and to facilitate the creation and use of instrument models to develop the likelihood model.


Prior models
------------

The package includes a prior model for the sky’s brightness distribution across different wavelengths, which can be customized to meet user needs in both spatial and spectral dimensions.
This model allows for the generation of spatially uncorrelated point sources or spatially correlated extended sources, as described by the correlated field model in [1]_.

In the spectral dimension, the model can be a power law, describe the correlation structure of the logarithmic flux using a Wiener process along the spectral axis or combine both of these models. The prior model’s structure is designed to be flexible, allowing for modifications to accommodate  additional dimensions and correlation structures. A demo for the implementation of such prior models can be found at:

.. toctree::
        :maxdepth: 1

        spectral_sky_demo

Likelihood models
-----------------

**J-UBIK** implements several instrument models (Chandra, eROSITA, JWST) and their respective data-
and response-loading functionalities, enabling their seamless integration into the inference pipeline.
**J-UBIK** is not only capable of reconstructing signals from real data; since each instrument model acts as a digital twin of the corresponding instrument, it can also be used to generate simulated data by passing sky prior models through the instrument’s response.
The package includes demos for the X-ray observatories Chandra and eROSITA which illustrate how to use or build these models.

.. toctree::
        :maxdepth: 1

        chandra_likelihood_demo


Inference with J-UBIK and NIFTy.re:
-----------------------------------

**J-UBIKs** prior and likelihood models can be user for inference using **NIFTy.re**. In the following demo we showcase how to construct an inference pipeline to obtain posterior estimates.


.. [1] P. Arras, P. Frank, P. Haim, J. Knollmüller, R. Leike, M. Reinecke, and T. Enßlin (2022), “Variable structures in M87 from space, time and frequency resolved interferometry,”*** Nature Astronomy, 6(2), 259–269. [DOI:10.1038/s41550-021-01548-0] <https://doi.org/10.1038/s41550-021-01548-0>_
