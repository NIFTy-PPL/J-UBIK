import nifty8.re as jft
from ..jwst.parametric_model import build_parametric_prior

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from functools import reduce


class ColorMixer(jft.Model):
    def __init__(
        self,
        diagonal: jft.Model,
        off_diagonal: jft.Model,
        components_domain: jft.ShapeWithDtype
    ):
        self._dia = diagonal
        self._off = off_diagonal
        self._upper_triangle = np.triu_indices(components_domain.shape[0], k=1)

        super().__init__(
            domain=diagonal.domain | off_diagonal.domain,
            target=components_domain
        )

    def matrix(self, parameters: dict):
        dia = self._dia(parameters)
        off = self._off(parameters)

        # outer
        mat = (dia[:, None] * dia[None, :])
        # FIXME: THIS IS ONLY DONE TO UNDERSTAND THE PRIORS
        mat = jnp.sqrt(mat)
        mat = mat.at[self._upper_triangle].set(
            mat[self._upper_triangle] * off)
        mat = mat.at[self._upper_triangle[1], self._upper_triangle[0]].set(
            mat[self._upper_triangle])
        return mat

    def __call__(self, parameters: dict, components: ArrayLike):
        mat = self.matrix(parameters)
        return jnp.einsum('ij,jkl->ikl', mat, components)


def build_colormix(
        prefix,
        components_target,
        diagonal_prior,
        off_diagonal_prior
):
    assert len(components_target.shape) == 3

    n = components_target.shape[0]
    print(f'Building {n} frequency components')

    shape_dia = n
    shape_off = (n**2 - n) // 2

    prefix = f'{prefix}_colormix'

    diagonal = jft.Model(
        build_parametric_prior(prefix + '_dia', diagonal_prior, shape_dia),
        domain={prefix + '_dia': jft.ShapeWithDtype(shape_dia)})
    off_diagonal = jft.Model(
        build_parametric_prior(prefix + '_off', off_diagonal_prior, shape_off),
        domain={prefix + '_off': jft.ShapeWithDtype(shape_off)})

    return ColorMixer(diagonal, off_diagonal, components_target)


class Components(jft.Model):
    def __init__(self, components: list[jft.Model], out_shape: tuple[int]):
        assert len(out_shape) == 2

        self._comps = components
        self._out_shape = out_shape

        cdomain = reduce(lambda x, y: x | y, [c.domain for c in self._comps])
        super().__init__(domain=cdomain)

    def __call__(self, x):
        return jnp.array([c(x)[:self._out_shape[0], :self._out_shape[1]]
                          for c in self._comps])


class ColorMixComponents(jft.Model):
    def __init__(self, components: Components, colormix: ColorMixer):
        self.components = components
        self.color = colormix

        super().__init__(domain=self.components.domain | self.color.domain)

    def __call__(self, x):
        comps = self.components(x)
        return jnp.exp(self.color(x, comps))


def component(key, shape, distances, config):
    cfm = jft.CorrelatedFieldMaker(prefix=key)
    cfm.set_amplitude_total_offset(**config['offset'])
    cfm.add_fluctuations(shape, distances, **config['fluctuations'])
    return cfm.finalize()


def build_components(
    prefix: str,
    shape: tuple[int],
    distances: tuple[int],
    padding_ratio: float,
    prior_config: dict
):

    prefix = f'{prefix}_comp_'

    pad_shape = [int(s*padding_ratio) for s in shape]

    components = []
    for key, config in prior_config.items():
        components.append(
            component(f'{prefix}_{key}', pad_shape, distances, config))

    return Components(components, shape)


def build_colormix_components(
    prefix: str,
    colormix_config: dict,
    components_config: dict
):

    comps = build_components(
        prefix=prefix,
        shape=components_config['shape'],
        distances=components_config['distances'],
        padding_ratio=components_config['s_padding_ratio'],
        prior_config=components_config['prior'])

    color = build_colormix(
        prefix,
        comps.target,
        diagonal_prior=colormix_config['diagonal_prior'],
        off_diagonal_prior=colormix_config['off_diagonal_prior'])

    return ColorMixComponents(comps, color)


def prior_samples_colormix_components(sky_model, n_samples=4):
    from jax import random
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    key = random.PRNGKey(42)
    N_comps = len(sky_model.components._comps)

    for _ in range(n_samples):
        key, rec_key = random.split(key, 2)
        x = jft.random_like(key, sky_model.domain)

        comps = sky_model.components(x)
        correlated_comps = sky_model(x)

        mat_mean = sky_model.color.matrix(x)
        print()
        print('Color Mixing Matrix')
        print(mat_mean)
        print()

        fig, axes = plt.subplots(N_comps, 2)
        for ax, cor_comps, comps in zip(axes, correlated_comps, comps):
            im0 = ax[0].imshow(cor_comps, origin='lower', norm=LogNorm())
            im1 = ax[1].imshow(np.exp(comps), origin='lower', norm=LogNorm())
            plt.colorbar(im0, ax=ax[0])
            plt.colorbar(im1, ax=ax[1])
            ax[0].set_title('Correlated Comps')
            ax[1].set_title('Comps')

        plt.show()
