Examples: Inference pipelines
=============================
These steps can be plugged together for different datasets or observations and even different instruments to build a pipeline. The examples below show pipelines for Chandra, eRosita and JWST.
In particular, these demos showcase ways to build pipelines using yaml files, which is an optional feature. For the beginning we recommend to look at the prior models in detail to get a better understading for the different parameters, especially looking at prior samples for diverse set of hyperparameters.
The structure of the yaml files is dependent on the used instrument(s) and prior models. Further explanation can be found in the "YAML Configuration File Structure" and the "Sky Model" sections of the pipeline demos.

.. toctree::
        :maxdepth: 1

        chandra_demo
        erosita_demo
        jwst_demo


