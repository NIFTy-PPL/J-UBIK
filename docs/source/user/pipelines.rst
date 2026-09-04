Examples: Inference pipelines
=============================
These steps can be plugged together for different datasets or observations and even different instruments to build a pipeline. The examples below show pipelines for Chandra, eROSITA, JWST, and a synthetic radio-interferometric observation using Resolve.
The Chandra, eROSITA, and JWST demos also showcase ways to build pipelines using YAML files, which is an optional feature. Before building a pipeline, we recommend exploring the prior models and drawing prior samples for several sets of hyperparameters.
The structure of the yaml files is dependent on the used instrument(s) and prior models. Further explanation can be found in the "YAML Configuration File Structure" and the "Sky Model" sections of the pipeline demos.

.. toctree::
        :maxdepth: 1

        chandra_demo
        erosita_demo
        jwst_demo
        resolve_synthetic_demo
