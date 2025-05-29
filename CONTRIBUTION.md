# Contribution to J-UBIK

There are several ways to contribute to J-UBIK.
- start or contribute to [discussions](https://github.com/NIFTy-PPL/J-UBIK/discussions/)
- raising [issues](https://github.com/NIFTy-PPL/J-UBIK/issues) if you notice some possible bug.
- contribute to the codebase by forking the repository and opening a [pull request](https://github.com/NIFTy-PPL/J-UBIK/pulls).

You may consider checking if there is an existing issue or discussion for your problem or need before opening a pull request.

## Contribution Guidelines

We welcome discussions and contributions for new features and instrument implementations!
If you would like to implement a new instrument for use within the J-UBIK inference pipeline, please ensure that the most important instrument effects are properly modeled (this includes, e.g., the point spread function (PSF), exposure, and effective area).

The next step is implement a function which maps the physical field (e.g. the photon flux field) to the measured quantity (e.g. photon-count data) using a differentiable code library such as [JAX](https://github.com/jax-ml/jax).
This allows [NIFTy](https://github.com/NIFTy-PPL/NIFTy) (**N**umerical **I**nformation **F**ield **T**heor**y**) to use `JAX` automatic differentiation for its inference algorithms.
