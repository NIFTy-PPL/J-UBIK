What is Information Field Theory?
=================================

Information Field Theory (IFT) is a theoretical Bayesian framework for inferring continuous physical fields from finite datasets. At the core of IFT is Bayes’ theorem:

.. math::
    \mathcal{P}(s|d) \propto \mathcal{P}(d|s) \mathcal{P}(s) ,

where the prior :math:`\mathcal{P}(s)` represents our knowledge about the signal $s$ before observing the data :math:`d`, and the likelihood :math:`\mathcal{P}(d|s)` describes the measurement process. The posterior :math:`\mathcal{P}(s|d)` is the primary measure of interest in the inference process.

The resources listed below introduce the theoretical foundations of IFT and demonstrate its applications, including implementations with the J-UBIK software package.


.. tip:: For an introduction to information field theory, see [1]_. Applications of IFT and the J-UBIK package to X-ray data can be found at [2], [3] and [4].


.. [1] T.A. Enßlin (2019), "Information theory for fields", Annalen der Physik; `[DOI] <https://doi.org/10.1002/andp.201800127>`_, `[arXiv:1804.03350] <https://arxiv.org/abs/1804.03350>`_

.. [2] M. Westerkamp, V. Eberle, M. Guardiani, P. Frank, L. Scheel-Platz, P. Arras, J. Knollmüller, J. Stadler, and T. Enßlin (2024), “The first spatio-spectral Bayesian imaging of SN1006 in X-rays,” Astronomy & Astrophysics, Vol. 684, A155, EDP Sciences. [DOI:10.1051/0004-6361/202347750] <http://dx.doi.org/10.1051/0004-6361/202347750>_

.. [3] V. Eberle, M. Guardiani, and M. Westerkamp (2025), “Bayesian Multiband Imaging of SN1987A in the Large Magellanic Cloud with SRG/eROSITA,” Zenodo. [DOI:10.5281/zenodo.16918521] <https://doi.org/10.5281/zenodo.16918521>_

.. [4] M. Guardiani, V. Eberle, M. Westerkamp, J. Rüstig, P. Frank, and T. Enßlin (2025), “Latent-space Field Tension for Astrophysical Component Detection: An application to X-ray imaging,” arXiv preprint. [arXiv:2506.20758] <https://arxiv.org/abs/2506.20758>_


