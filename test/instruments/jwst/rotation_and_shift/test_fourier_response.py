import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u

pytest.importorskip("jax_finufft")

from jubik.instruments.jwst.parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)
from jubik.instruments.jwst.parse.rotation_and_shift.rotation_and_shift import (
    NufftConfig,
)
from jubik.instruments.jwst.rotation_and_shift.coordinates_correction import (
    Coordinates,
    CoordinatesCorrectedShiftOnly,
    ShiftAndRotationCorrection,
)
from jubik.instruments.jwst.rotation_and_shift.fourier_response import (
    build_fourier_response,
)

SKY = (24, 28)
S = 3  # kernel pixels per detector pixel
OUT = (11, 11)  # detector pixels, more than the sky
ONE = u.Quantity([1.0, 1.0], u.arcsec)  # sky pixel scales: band in cycles / pixel
# per exposure: rotation (deg), detector pixel in sky pixels along its two axes,
# kernel std and centre offset (kernel pixels, row and column), shift latent
EXPOSURES = [
    (20.8, (3.75, 3.75), (2.0, 2.4), (0.3, -0.2), (0.7, -1.3)),
    (-10.0, (3.7, 3.8), (2.2, 2.0), (-0.4, 0.1), (-0.6, 2.4)),
]
# sky pixels: the second exposure moves by 19, more than the padding without the
# prior's reach (3 sigma) leaves the PSF's core
SHIFT_SIGMA = 8.0


def _jacobian(theta, scale):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array([[c, -s], [s, c]]) * np.array(scale)  # columns: detector axes


def _centres(jac, out=OUT):
    """Subsampled pixel centres on the sky grid, (2, H S, W S), centred on the sky."""
    sub = [(np.arange(n * S) + 0.5) / S - 0.5 - (n - 1) / 2 for n in out]
    det = np.stack(np.meshgrid(*sub, indexing="ij"))
    return (
        np.einsum("ij,j...->i...", jac, det) + ((np.array(SKY) - 1) / 2)[:, None, None]
    )


def _kernel(std, offset, size=31):
    m = np.arange(size) - (size - 1) / 2
    g = np.exp(-0.5 * ((m[:, None] - offset[0]) / std[0]) ** 2)
    g = g * np.exp(-0.5 * ((m - offset[1]) / std[1]) ** 2)
    return g / g.sum()


def _shift_coordinates(fixed):
    prior = CoordinatesCorrectionPriorConfig.from_yaml_dict(
        {
            "model": "shift",
            "shift": ["normal", 0.0, SHIFT_SIGMA],
            "rotation": ["delta", 0.0, 0.0],
            "shift_unit": "arcsec",
            "rotation_unit": "deg",
        },
        shift_shape=(len(fixed), 2),
        rotation_shape=(len(fixed), 1),
    )
    correction = ShiftAndRotationCorrection("f", prior, rotation_center=None)
    return CoordinatesCorrectedShiftOnly(correction, fixed, ONE, (Ellipsis, None, None))


def _exact(sky, centre, jac, std, offset, shift, nodes=24):
    """sum_n sky_n times the pixel average of the PSF at the pixel minus sky pixel n,
    the PSF a Gaussian in sky pixels; Gauss-Legendre over the pixel."""
    t, w = np.polynomial.legendre.leggauss(nodes)
    t, w = t / 2, w / 2
    u = np.stack(np.meshgrid(t, t, indexing="ij")).reshape(2, -1)  # detector px
    ww = np.outer(w, w).ravel()
    cov = (jac / S) @ np.diag(np.square(std)) @ (jac / S).T
    icov, norm = np.linalg.inv(cov), 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    n = np.stack(np.meshgrid(*map(np.arange, SKY), indexing="ij")).reshape(2, -1)
    out = []
    for x in centre.T:  # (2,) per pixel
        q = (x + shift - jac @ np.array(offset) / S)[:, None] + jac @ u
        d = q[:, :, None] - n[:, None, :]  # (2, nodes, sky)
        g = norm * np.exp(-0.5 * np.einsum("iqs,ij,jqs->qs", d, icov, d))
        out.append(ww @ g @ sky.ravel())
    return np.array(out)


def test_matches_the_exact_pixel_integral():
    rng = np.random.default_rng(7)
    sky = rng.uniform(0, 1, SKY)  # flux up to the edges: a wrap-around would show
    jacs = [_jacobian(th, sc) for th, sc, *_ in EXPOSURES]
    fixed = np.stack([_centres(j) for j in jacs])
    kernels = np.stack([_kernel(std, off) for _, _, std, off, _ in EXPOSURES])
    coordinates = _shift_coordinates(fixed)
    centres = fixed.reshape(2, 2, OUT[0], S, OUT[1], S).mean((3, 5))
    mask = np.all((centres > -0.5) & (centres < np.array(SKY)[:, None, None] - 0.5), 1)
    # 0.38 cycles per sky pixel: below the kernel grid's Nyquist, where the
    # narrowest PSF (2.5 sky pixels) passes 2e-8
    response = build_fourier_response(
        {"sky": jft.ShapeWithDtype(SKY, float)},
        coordinates,
        kernels,
        S,
        NufftConfig(band_limit=0.38),
        ONE,
        mask,
    )
    latent = np.array([e[-1] for e in EXPOSURES])
    out = np.asarray(response({"sky": sky, "f_shift": latent}))
    # the shift's derivative (a phase derivative) against central differences
    model = lambda xi: response({"sky": sky, "f_shift": xi})  # noqa: E731
    direction, h = np.array([[0.3, -1.0], [0.8, 0.5]]), 1e-4
    _, tangent = jax.jvp(model, (latent,), (direction,))
    diff = (model(latent + h * direction) - model(latent - h * direction)) / (2 * h)
    np.testing.assert_allclose(tangent, diff, rtol=0, atol=1e-6)
    assert out.shape == (2,) + OUT
    for e, (jac, (_, _, std, off, xi)) in enumerate(zip(jacs, EXPOSURES)):
        exact = _exact(
            sky, centres[e][:, mask[e]], jac, std, off, SHIFT_SIGMA * np.array(xi)
        )
        np.testing.assert_allclose(out[e][mask[e]], exact, rtol=0, atol=1e-7)


# production kernels are even (STPSF 168 x 168): recentred onto an odd grid
@pytest.mark.parametrize("size", [31, 30])
def test_without_correction_and_one_kernel(size):
    # no shift latents; one 2-D kernel for every exposure; no mask, and pixels 22
    # sky pixels past the sky, beyond the kernel's spare reach: the padding must
    # hold the data's extent too
    jac, wide = _jacobian(20.8, (3.75, 3.75)), (15, 15)
    fixed = np.stack([_centres(jac, wide)] * 2)
    sky = np.random.default_rng(8).uniform(0, 1, SKY)
    response = build_fourier_response(
        {"sky": jft.ShapeWithDtype(SKY, float)},
        Coordinates(fixed),
        _kernel((2.0, 2.0), (0.0, 0.0), size),
        S,
        NufftConfig(band_limit=0.38),
        ONE,
    )
    assert response.domain.keys() == {"sky"}
    out = np.asarray(response({"sky": sky}))
    np.testing.assert_allclose(out[0], out[1], rtol=0, atol=1e-12)
    centres = fixed[0].reshape(2, wide[0], S, wide[1], S).mean((2, 4))
    exact = _exact(sky, centres.reshape(2, -1), jac, (2.0, 2.0), (0, 0), np.zeros(2))
    np.testing.assert_allclose(out[0].ravel(), exact, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    "config, match",
    [
        (NufftConfig(), "band_limit"),
        (NufftConfig(band_limit=0.3, mode="wrap"), "constant"),
        (NufftConfig(band_limit=0.45), "Nyquist"),  # 0.45 * 3.75 / 3 > 0.5
    ],
)
def test_invalid_configs_raise(config, match):
    fixed = np.stack([_centres(_jacobian(0.0, (3.75, 3.75)))])
    with pytest.raises(ValueError, match=match):
        build_fourier_response(
            {"sky": jft.ShapeWithDtype(SKY, float)},
            Coordinates(fixed),
            _kernel((2.0, 2.0), (0.0, 0.0)),
            S,
            config,
            ONE,
        )
