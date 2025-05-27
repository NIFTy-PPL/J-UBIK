# Contribution to J-UBIK

There are several ways to contribute to J-UBIK.
- start or contribute to [discussions](https://github.com/NIFTy-PPL/J-UBIK/discussions/)
- raising [issues](https://github.com/NIFTy-PPL/J-UBIK/issues) if you notice some possible bug.
- contribute to the codebase by forking the repository and opening a [pull request](https://github.com/NIFTy-PPL/J-UBIK/pulls).

You may consider, checking if there is an existing issue or discussion for your problem or need before opening a pull request.

## Contribution Guidelines

We are open for discussion and happy to help implementing new feautures and instruments.
If you want to implement a new instrument, in order to use it in your inference pipeline within J-UBIK you need to take care of the most important instrument effects. (e.g. PSF, exposure, effective area). In the next step you need to able to build a function which maps the physical field to the measured quantity using a differentiable code library like [JAX](https://github.com/jax-ml/jax). This is needed, so that [NIFTy](https://github.com/NIFTy-PPL/NIFTy), the package for **N**umerical **I**nformation **F**ield **T**heor**y** and its inference algorithms, can use JAX aufodif.