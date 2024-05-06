# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Julian Ruestig


from jax.scipy.signal import convolve2d, fftconvolve
from functools import partial

from typing import Callable
from numpy.typing import ArrayLike


def PsfOperator_dir(field, kernel):
    '''Creates a Psf-operator: convolution of field by kernel'''
    return convolve2d(field, kernel, mode='same')


def PsfOperator_fft(field, kernel):
    '''Creates a Psf-operator: convolution of field by kernel'''
    return fftconvolve(field, kernel, mode='same')


def instantiate_psf(psf: ArrayLike | None) -> Callable:
    if psf is None:
        return lambda x: x

    return partial(PsfOperator_fft, kernel=psf)
