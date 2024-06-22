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

    from charm_lensing.models.parametric_models import build_parametric
    from charm_lensing.spaces import get_xycoords

    prefix = f'{prefix}_comp_'

    pad_shape = [int(s*padding_ratio) for s in shape]
    coords = get_xycoords(pad_shape, distances)

    components = []
    for key, config in prior_config.items():
        comp = component(f'{prefix}_{key}', pad_shape, distances, config)

        if 'mean' in config:
            mean = build_parametric(
                coords=coords,
                prefix=f'{prefix}_{key}',
                model_config=config['mean'],
            )
            model = jft.Model(
                lambda x: jnp.log(mean(x)),  # + comp(x),
                domain=mean.domain | comp.domain)

        else:
            model = comp

        components.append(model)

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
