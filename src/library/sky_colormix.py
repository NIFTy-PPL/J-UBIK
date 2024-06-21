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

    def _matrix(self, parameters: dict):
        dia = self._dia(parameters)
        off = self._off(parameters)

        # outer
        mat = (dia[:, None] * dia[None, :])
        mat = mat.at[self._upper_triangle].set(
            mat[self._upper_triangle] * off)
        mat = mat.at[self._upper_triangle[1], self._upper_triangle[0]].set(
            mat[self._upper_triangle])
        return mat

    def __call__(self, parameters: dict, components: ArrayLike):
        mat = self._matrix(parameters)
        return jnp.einsum('ij,jkl->ikl', mat, components)


class Components(jft.Model):
    def __init__(self, components: list[jft.Model]):
        self._comps = components
        cdomain = reduce(lambda x, y: x | y, [c.domain for c in self._comps])
        super().__init__(domain=cdomain)

    def __call__(self, x):
        return jnp.array([c(x) for c in self._comps])


class ColorMixComponents(jft.Model):
    def __init__(self, components: Components, colormix: ColorMixer):
        self._comps = components
        self._color = colormix

        super().__init__(domain=self._comps.domain | self._color.domain)

    def __call__(self, x):
        comps = self._comps(x)
        return self._color(x, comps)


def build_colormix(
        prefix,
        components_target,
        diagonal_prior,
        off_diagonal_prior
):
    assert len(components_target.shape) == 3

    n = components_target.shape[0]
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


def build_components(prefix, dims, distances, configs):
    prefix = f'{prefix}_comp'

    def component(key, dims, distances, config):
        cfm_model_maker = jft.CorrelatedFieldMaker(f'{key}_')
        cfm_model_maker.add_fluctuations(
            dims, distances, **config['fluctuations'])
        cfm_model_maker.set_amplitude_total_offset(**config['amplitude'])
        return cfm_model_maker.finalize()

    return Components([component(f'{prefix}_{key}', dims, distances, config)
                       for key, config in configs.items()])


def build_colormix_components(
        prefix: str,
        colormix_config: dict,
        components_config: dict
):

    comps = build_components(
        prefix,
        components_config['dims'],
        components_config['distances'],
        components_config['prior'])

    color = build_colormix(
        prefix,
        comps.target,
        diagonal_prior=colormix_config['diagonal_prior'],
        off_diagonal_prior=colormix_config['off_diagonal_prior'])

    return ColorMixComponents(comps, color)
