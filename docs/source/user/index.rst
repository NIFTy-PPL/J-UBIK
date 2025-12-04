J-UBIK User Guide
=================

This guide is an overview and explains the main conceptual idea behind J-UBIK (JAX-accelerated Universal Bayesian Imaging Kit for X-ray), NIFTy.re and IFT.

The foundation of J-UBIK and NIFTy.re is the theoretical framework of information field theory (IFT), which applies Bayesian inference to address the underconstrained problem of reconstructing continuous physical fields from finite datasets.

However, J-UBIK is more driven by real instrument and their data. It facilitates the implementation of the central components of Bayesian inference, namely likelihood models for different instruments. It currently includes implementations for X-ray telescopes (Chandra and eROSITA) as well as the infrared observatory JWST. Several demos show how different likelihood implementations (Chandra & eROSITA):

Further information on these topics, as well as on the role of prior and likelihood models in Bayesian inference, can be found here:

.. toctree::
        :maxdepth: 1
        
        ift
        nifty
        jubik


More details on the API and the demos for the different instruments can be found at the `API reference <../mod/jubik.html>`_.

Details on how to cite the J-UBIK package can be found at

.. toctree::
        :maxdepth: 1

        citations
