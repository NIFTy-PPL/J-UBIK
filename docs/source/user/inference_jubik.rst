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