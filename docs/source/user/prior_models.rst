Prior models
============

The package includes a prior model for the sky’s brightness distribution across different wavelengths, which can be customized to meet user needs in both spatial and spectral dimensions.
This model allows for the generation of spatially uncorrelated point sources or spatially correlated extended sources, as described by the correlated field model in [1]_.

In the spectral dimension, the model can be a power law, describe the correlation structure of the logarithmic flux using a Wiener process along the spectral axis or combine both of these models. The prior model’s structure is designed to be flexible, allowing for modifications to accommodate additional dimensions and correlation structures. A demo for the implementation of such prior models can be found at:

.. toctree::
        :maxdepth: 1

        spectral_sky_demo
        point_source_sky_demo

However, you can include your own prior models using NIFTy. For more information read the previous page

.. toctree::
        :maxdepth: 1
        
        nifty


.. [1] Arras P., Frank P., Haim P., Knollmüller J., Leike R., Reinecke M., and Enßlin T. (2022), “Variable structures in M87 from space, time and frequency resolved interferometry,” Nature Astronomy, 6(2), 259–269. `[DOI] <https://doi.org/10.1038/s41550-021-01548-0>`_
