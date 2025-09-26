What is Information Field Theory?
=================================

Information Field Theory (IFT) is a theoretical Bayesian framework for inferring continuous physical fields from finite datasets. At the core of IFT is Bayes’ theorem:

.. math::
    \mathcal{P}(s|d) \propto \mathcal{P}(d|s) \mathcal{P}(s) ,

where the prior :math:`\mathcal{P}(s)` represents our knowledge about the signal $s$ before observing the data :math:`d`, and the likelihood :math:`\mathcal{P}(d|s)` describes the measurement process. The posterior :math:`\mathcal{P}(s|d)` is the primary measure of interest in the inference process.

The resources listed below introduce the theoretical foundations of IFT and demonstrate its applications, including implementations with the J-UBIK software package.


.. tip:: For an introduction to information field theory, see [#ift]_. Applications of IFT and the J-UBIK package to X-ray data can be found at [#west]_, [#lmc]_ and [#latent]_.


.. [#ift] Enßlin T. (2019), "Information theory for fields", Annalen der Physik; `[DOI] <https://doi.org/10.1002/andp.201800127>`_, `[arXiv:1804.03350] <https://arxiv.org/abs/1804.03350>`_
.. [#west] Westerkamp M., Eberle V., Guardiani M., Frank P., Scheel-Platz L., Arras P., Knollmüller J., Stadler J., and Enßlin T. (2024), "The first spatio-spectral Bayesian imaging of SN1006 in X-rays," Astronomy & Astrophysics, Vol. 684, A155, EDP Sciences; `[DOI] <http://dx.doi.org/10.1051/0004-6361/202347750>`_
.. [#lmc] Eberle V., Guardiani M., and Westerkamp M. (2025), "Bayesian Multiband Imaging of SN1987A in the Large Magellanic Cloud with SRG/eROSITA," Zenodo. `[DOI] <https://doi.org/10.5281/zenodo.16918521>`_
.. [#latent] Guardiani M., Eberle V., Westerkamp M., Rüstig J., Frank P., and Enßlin T. (2025), "Latent-space Field Tension for Astrophysical Component Detection: An application to X-ray imaging," arXiv preprint. `[arXiv:2506.20758] <https://arxiv.org/abs/2506.20758>`_
