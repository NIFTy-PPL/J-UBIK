---
title: 'Universal Bayesian Imaging Kit or X-ray Astronomy'
tags:
  - Python
  - Astronomy
  - Imaging
  - Gaussian Processes
  - Variational Inference
authors:
  - name: Vincent Eberle
    orcid: 0000-0002-5713-3475
    affiliation: "1, 2"
  - name: Matteo Guardiani
    orcid: 0000-0002-4905-6692
    affiliation: "1, 2"
  - name: Margret Westerkamp
    orcid: 0000-0001-7218-8282
    affiliation: "1, 2"
  - name: Philipp Frank
    orcid: 0000-0001-5610-3779
    affiliation: "1"
  - name: Torsten A. Enßlin
    orcid: 0000-0001-5246-1624
    affiliation: "1, 2"
affiliations:
  - name: Max Planck Institute for Astrophysics, Karl-Schwarzschild-Straße 1, 85748 Garching bei München, Germany
    index: 1
  - name: Ludwig Maximilian University of Munich, Geschwister-Scholl-Platz 1, 80539 München, Germany
    index: 2
  - name: Department of Astrophysics, University of Vienna, Türkenschanzstraße 17, A-1180 Vienna, Austria
date: 2 September 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

<!--
## JAX + NIFTy Paper
* USP: selling point: speed
* Bonus: higher order diff for more efficient optimization and all of Tensorflow and Tensorflow for all
* GP
  * Regular Grid Refinement
  * KISS-GP
  * Grid Refinement
* Posterior Approx.
  * HMC but with variable dtype handling
  * JIT-able VI and also (indirectly) available for Tensorflow
* predecessor enabled 100B reconstruction
* middle ground between tools like blackjax and pymc
-->

# Summary
To facilitate multi-instrument analysis of correlated signals in general, we are developing the Universal Bayesian Imaging Kit (UBIK), a flexible and modular framework for high-fidelity Bayesian imaging. UBIK is based on the NIFTy.re software, which is an accelerated 
Bayesian framework for imaging. It facilitates the implementation of the main components of Bayesian inference, i.e. likelihood models for different instruments and prior models. The package includes three instrument implementations, two X-ray telescopes, Chandra and eROSITA, and JWST in the infrared, as well as a prior model adaptable to different sky realisations. The demos show how the likelihood and prior implementation can be integrated into an inference pipeline, with the possibility to choose different optimisation schemes such as maximum a posteriori or variational inference.


# Statement of Need

# Past and ongoing research projects

# Conclusion


# Acknowledgements

# More structure info here
https://joss.readthedocs.io/en/latest/paper.html#what-should-my-paper-contain

# References

<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"
# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }
-->
