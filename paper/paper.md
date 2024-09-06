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
To facilitate multi-instrument analysis of correlated signals in general, we are developing 
the Universal Bayesian Imaging Kit (UBIK), a flexible and modular framework for high-fidelity 
Bayesian imaging. UBIK is based on the NIFTy.re [@Edenhofer:2024] software, which is an accelerated 
Bayesian framework for imaging. It facilitates the implementation of the main components of Bayesian 
inference, i.e. likelihood models for different instruments and prior models. The package includes
three instrument implementations, two X-ray telescopes, Chandra and eROSITA, and JWST in the infrared,
as well as a prior model adaptable to different sky realisations. The demos show how the likelihood 
and prior implementation can be integrated into an inference pipeline, with the possibility to
choose different optimisation schemes such as maximum a posteriori or variational inference.

# Statement of Need
In imaging , we are often confronted with high-dimensional signals of interest,
that vary in terms of space, time and energy. In astronomy, for example, the new generation 
of telescopes  offers many opportunities in capturing those signals, 
but at the same time also challenges in imaging,
to get the most information out of the corresponding data. 
These challenges include modelling the response of the instruments to the signal, 
the modelling of the noise structure and the modelling of the signal, which is typically
a mixture of overlapping signal components with non-trivial correlation structures that need 
to be separated.

UBIK as the universal Bayesian imaging kit, uses Bayesian statistics to facilitate the 
reconstruction of these complex signals, whether in astronomy or in other areas such as medical imaging,
from multi-instrument data. UBIK is based on the theory of information field theory [@Ennslin:2013]
and on the according software package NIFTy.re [@Edenhofer:2024], which is an accelerated and
jax-bases version of NIFTy [@Arras:2019]. According to this, it uses a prior model
, which describes the prior assumptions before we have any further knowledge from 
the instruments data, in a generative fashion and a likelihood model, which describes the measurement 
or in other words the responses of the possible multiple instruments and the noise statistics.
Using NIFTy.re as a basis, the package supports to give the physical, high-dimensional
signal field a sparse, adaptive and distributed representations and provides different methods,
like MAP, HMC or two variational inference algorithms, MGVI [@Knollmueller:2020] and 
geoVI [@Frank:2021], for
the efficient inference using parallel computing on clusters and GPUs. 

Up to now, the likelihood models and prior models have been built from scratch for different 
imaging problems that Nifty.re tackles. This accounts for most of the work in this Bayesian imaging
process. UBIK approaches this difficulty from two sides. On the one hand, UBIK contains tools to
facilitate the implementation of new likelihood and prior models. It is a toolbox that allows for
different types of response applications, for example using spatially variant or invariant psfs,
allowing for different types of noise statistics of the signal, such as Poissonian or Gaussian, 
and allowing the user to build different correlation structures on different components of the sky.
On the other hand, it includes a set of instrument implementations. So far, three instrument 
implementations are accessible, i.e. Chandra, eROSITA and JWST, and we expect this number to 
grow with the number of users, leading to a set of easily accessible inference algorithms 
for different instruments. Ultimately UBIK enables the user, through Bayesian 
statistics, not only to obtain posterior samples and hence measures of interest such as the
posterior mean and uncertainty of the signal for a several data sets, but also to 
perform multi-instrument reconstructions.

The according software has been applied already in [@Westerkamp:2023] and currently according publications
on eROSITA and JWST are in perparation. Afterwards, the set of instruments even further 
by already exitsing imaging tasks with NIFTy and NIFTy.re like, [@Platz:2023], ...., and new ones.

# Bayesian Imaging with UBIK
The basis of the UBIK package is Bayes theorem, 
$$ \mathcal{P}(s|d) \propto \mathcal{P}(d|s)\mathcal{P}(s),$$
where the prior $\mathcal{P}(s)$ describes the knowledge on the signal, $s$, before the data, 
$d$, is given, the likelihood $\mathcal{P}(d|s)$ describes the measurement and the actual 
or an approximation of the posterior $\mathcal{P}(s|d)$ is the measure of interest in 
the inference. The main task UBIK shall be used for, is to model the prior in a generative fashion and to use or build
instrument models easily in order to get a likelihood model. The package itself contains
demos for Chandra, eROSITA and JWST, which shows how to use or build this models and how to 
generate an inference pipeline upon on that to get posterior estimates.

## Prior models
## Likelihood models


# Conclusion


# Acknowledgements


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
