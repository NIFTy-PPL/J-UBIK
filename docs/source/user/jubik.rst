What is J-UBIK?
===============

**J-UBIK** is built on top of **NIFTy.re**, a software package designed for the accelerated Bayesian inference of physical fields. Currently, the scope of application is limited to a few astrophysical instruments, but there are plans to expand the areas of application further. The main role of **J-UBIK** is to supply with user with prior models, which are built in a generative fashion and to facilitate the creation and use of instrument models to develop the likelihood model. **J-UBIK** implements several instrument models (Chandra, eROSITA, JWST) and their respective data-
and response-loading functionalities, enabling their seamless integration into the inference pipeline. **J-UBIK** is not only capable of reconstructing signals from real data; since each instrument model acts as a digital twin of the corresponding instrument, it can also be used to generate simulated data by passing sky prior models through the instrument’s response.

In following you'll get more information about:

.. toctree::
        :maxdepth: 1

        prior_models

If you want to learn more about inference with **J-UBIK**:

.. toctree::
        :maxdepth: 1

        inference_jubik

Once you are used to the behaviour of prior parameters and the instrument models encoding the likelihood, you can learn more about pipelines in the following:

.. toctree::
        :maxdepth: 1

        pipelines



