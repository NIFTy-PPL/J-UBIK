import nifty8.re as jft
from ..jwst.parametric_model import build_parametric_prior

from jax import vmap
import jax.numpy as jnp
import numpy as np

from typing import Optional

from functools import reduce


class ColorMatrix(jft.Model):
    def __init__(
        self,
        diagonal: jft.Model,
        off_diagonal: jft.Model,
        number_of_components: int,
        reference_color: Optional[int] = None,
    ):
        self._dia = diagonal
        self._off = off_diagonal
        self._upper_triangle = np.triu_indices(number_of_components, k=1)
        self._ref_color = reference_color

        super().__init__(
            domain=diagonal.domain | off_diagonal.domain,
        )

    def __call__(self, parameters: dict):
        dia = self._dia(parameters)
        off = self._off(parameters)

        # outer
        mat = (dia[:, None] * dia[None, :])
        # FIXME: THE sqrt IS ONLY DONE TO UNDERSTAND THE PRIORS
        mat = jnp.sqrt(mat)
        mat = mat.at[self._upper_triangle].set(mat[self._upper_triangle]*off)
        mat = mat.at[self._upper_triangle[1], self._upper_triangle[0]].set(
            mat[self._upper_triangle])
        # TODO: is this slowing down the execution or is this if else ignored
        # during jax.jit ???
        if self._ref_color is None:
            return mat
        mat = mat.at[:, self._ref_color].set(1.)
        return mat


class Components(jft.Model):
    def __init__(self, components: list[jft.Model], out_shape: tuple[int]):
        assert len(out_shape) == 2

        self.components = components
        self._out_shape = out_shape

        cdomain = reduce(
            lambda x, y: x | y, [c.domain for c in self.components])
        super().__init__(domain=cdomain)

    def __call__(self, x):
        return jnp.array([c(x)[:self._out_shape[0], :self._out_shape[1]]
                          for c in self.components])


class ColorMix(jft.Model):
    def __init__(self, components: Components, color_matrix: ColorMatrix):
        self.components = components
        self.color_matrix = color_matrix

        super().__init__(domain=self.components.domain | self.color_matrix.domain)

    def __call__(self, x):
        com = self.components(x)
        mat = self.color_matrix(x)
        return jnp.exp(jnp.einsum('ij,jkl->ikl', mat, com))


def build_color_matrix(
    prefix: str,
    number_of_components: int,
    diagonal_prior,
    off_diagonal_prior,
    reference_color: Optional[int] = None,
):

    n = number_of_components
    print(f'Building {n} frequency components')

    shape_dia = n
    shape_off = (n**2 - n) // 2

    prefix = f'{prefix}_matrix'

    diagonal = jft.Model(
        build_parametric_prior(prefix + '_dia', diagonal_prior, shape_dia),
        domain={prefix + '_dia': jft.ShapeWithDtype(shape_dia)})
    off_diagonal = jft.Model(
        build_parametric_prior(prefix + '_off', off_diagonal_prior, shape_off),
        domain={prefix + '_off': jft.ShapeWithDtype(shape_off)})

    return ColorMatrix(diagonal, off_diagonal, number_of_components, reference_color)
