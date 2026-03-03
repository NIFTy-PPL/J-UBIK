What is J-UBIK?
===============

**J-UBIK** is built on top of **NIFTy.re**, a software package designed for the accelerated Bayesian inference of physical fields. Currently, the scope of application is limited to a few astrophysical instruments, but there are plans to expand the areas of application further. The main role of **J-UBIK** is to supply with user with prior models, which are built in a generative fashion and to facilitate the creation and use of instrument models to develop the likelihood model. **J-UBIK** implements several instrument models (Chandra, eROSITA, JWST) and their respective data-
and response-loading functionalities, enabling their seamless integration into the inference pipeline. **J-UBIK** is not only capable of reconstructing signals from real data; since each instrument model acts as a digital twin of the corresponding instrument, it can also be used to generate simulated data by passing sky prior models through the instrument’s response.



Prior models
------------

The package includes a prior model for the sky’s brightness distribution across different wavelengths, which can be customized to meet user needs in both spatial and spectral dimensions.
This model allows for the generation of spatially uncorrelated point sources or spatially correlated extended sources, as described by the correlated field model in [1]_.

In the spectral dimension, the model can be a power law, describe the correlation structure of the logarithmic flux using a Wiener process along the spectral axis or combine both of these models. The prior model’s structure is designed to be flexible, allowing for modifications to accommodate additional dimensions and correlation structures. A demo for the implementation of such prior models can be found at:

.. toctree::
        :maxdepth: 1

        spectral_sky_demo
        point_source_sky_demo

However, you can include your own prior models using NIFTy. For more information read the previous page "What is NIFTy?".

Inference with J-UBIK and NIFTy:
--------------------------------

To do inference with different instrument, we need to load instrumental and observational data. The following demo shows how to do this for Chandra:

.. toctree::
        :maxdepth: 1

        chandra_likelihood_demo

This knowledge about the instrument and the measurement process encodes the likelihood. Combining the likelihood with the prior models described earlier, we can be build inference scripts using **NIFTy.re**. 


In the following demo we show how all the parts of J-UBIK work together: we get the observational and instrumental data, construct the instrument response function, define a prior model, define the likelihood and run inference to recover posterior samples modelling a Chandra observation of supernova remnant G299.2-2.9. 

.. toctree::
        :maxdepth: 1

        x-ray-imaging


Examples: Inference pipelines:
------------------------------
These steps can be plugged together for different datasets or observations and even different instruments to build a pipeline. The examples below show pipelines for Chandra, eRosita and JWST.
In particular, these demos showcase ways to build pipelines using yaml files, which is an optional feature. For the beginning we recommend to look at the prior models in detail to get a better understading for the different parameters, especially looking at prior samples for diverse set of hyperparameters.
The structure of the yaml files is dependent on the used instrument(s) and prior models. Further explanation can be found in the "YAML Configuration File Structure" and the "Sky Model" sections of the pipeline demos.

.. toctree::
        :maxdepth: 1

        chandra_demo
        erosita_demo
        jwst_demo


.. [1] Arras P., Frank P., Haim P., Knollmüller J., Leike R., Reinecke M., and Enßlin T. (2022), “Variable structures in M87 from space, time and frequency resolved interferometry,” Nature Astronomy, 6(2), 259–269. `[DOI] <https://doi.org/10.1038/s41550-021-01548-0>`_
