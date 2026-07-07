"""Orientation glyph for the roundtrip probes: a letter "F" on the sky.

The F is defined ONCE in physical sky offsets (East, North) in arcsec —
vertical stroke pointing North, arms pointing West — and rasterized by
each consumer in its own pixel convention:

- `rasterize_canonical` paints it in the canonical frame (dim0 = +Dec,
  dim1 = -RA; probes/README.md), where it must come out with the stroke
  along +dim0 and the arms along +dim1.  Under the house plot
  (origin="lower", East left) it reads as a normal F.
- Minting scripts paint the same `sample_points` through their data
  WCS, anchoring the truth to absolute sky positions.

An F is asymmetric under every rotation and reflection, so any frame
error is a unique element of the dihedral group D4.  `dihedral_verdict`
identifies it: it correlates an image against all eight D4 transforms
of the truth (shift-invariant via FFT cross-correlation) and returns
the winning transform — a correct roundtrip must return "identity".

Self-test:  uv run python probes/roundtrip/glyph.py
"""

from typing import Callable

import numpy as np

# The F in (East, North) arcsec offsets from the glyph anchor (stroke
# base).  Stroke North; both arms point West; asymmetric by design.
GLYPH_SEGMENTS = (
    ((0.0, 0.0), (0.0, 10.0)),    # stroke: 10" North
    ((0.0, 10.0), (-6.0, 10.0)),  # top arm: 6" West
    ((0.0, 5.0), (-4.0, 5.0)),    # middle arm: 4" West
)

DIHEDRAL = {
    "identity": lambda a: a,
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "transpose": lambda a: a.T,
    "anti-transpose": lambda a: np.rot90(a, 2).T,
    "flip-dim0": lambda a: a[::-1, :],
    "flip-dim1": lambda a: a[:, ::-1],
}


def sample_points(scale: float = 1.0, step: float = 0.25) -> np.ndarray:
    """Dense (East, North) arcsec samples along the glyph segments."""
    pts = []
    for (e0, n0), (e1, n1) in GLYPH_SEGMENTS:
        length = float(np.hypot(e1 - e0, n1 - n0)) * scale
        n_samples = max(int(np.ceil(length / step)) + 1, 2)
        t = np.linspace(0.0, 1.0, n_samples)
        pts.append(np.stack([
            (e0 + t * (e1 - e0)) * scale,
            (n0 + t * (n1 - n0)) * scale,
        ], axis=1))
    return np.concatenate(pts, axis=0)


def rasterize_canonical(
    shape: tuple[int, int],
    center_px: tuple[float, float],
    pixsize_arcsec: tuple[float, float],
    scale: float = 1.0,
) -> np.ndarray:
    """The glyph on a canonical-frame array (dim0=+Dec, dim1=-RA).

    Parameters
    ----------
    shape, center_px : canonical array shape and the (i, j) anchor pixel.
    pixsize_arcsec : (along dim0/Dec, along dim1/RA) pixel sizes.
    """
    sky = np.zeros(shape)
    for east, north in sample_points(scale):
        i = int(round(center_px[0] + north / pixsize_arcsec[0]))
        j = int(round(center_px[1] - east / pixsize_arcsec[1]))
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            sky[i, j] = 1.0
    return sky


def _shift_max_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Max over shifts of the normalized cross-correlation of a and b."""
    a = a - a.mean()
    b = b - b.mean()
    norm = np.sqrt((a**2).sum() * (b**2).sum())
    if norm == 0.0:
        return 0.0
    cc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    return float(cc.max() / norm)


def dihedral_verdict(
    image: np.ndarray, truth: np.ndarray
) -> tuple[str, dict[str, float]]:
    """Which D4 transform of `truth` the `image` matches best.

    Returns the winning transform name (a correct roundtrip returns
    "identity") and the full correlation table for reporting.
    """
    image = np.asarray(image, dtype=float)
    scores = {
        name: _shift_max_correlation(image, np.asarray(t(truth), dtype=float))
        for name, t in DIHEDRAL.items()
    }
    return max(scores, key=scores.get), scores


def _self_test() -> None:
    truth = rasterize_canonical((64, 64), (24.0, 36.0), (0.5, 0.5))
    assert truth.sum() > 40, "glyph rasterized too sparsely"

    # stroke North: topmost lit row is the arm row, far above the anchor
    lit = np.argwhere(truth > 0)
    assert lit[:, 0].max() == 24 + 20, "stroke does not extend +20 px North"
    # arms West: maximal-j lit pixels sit at the arm rows, j > anchor
    assert lit[:, 1].max() == 36 + 12, "arms do not extend +12 px (West)"

    for name, transform in DIHEDRAL.items():
        verdict, scores = dihedral_verdict(np.asarray(transform(truth)), truth)
        margin = scores[verdict] - max(
            v for k, v in scores.items() if k != verdict
        )
        assert verdict == name, f"{name} misidentified as {verdict}"
        assert margin > 0.05, f"{name}: weak margin {margin:.3f}"
        print(f"{name:15s} identified, margin {margin:.3f}")

    # smeared image (dirty-beam stand-in): convolve with a Gaussian
    from scipy.ndimage import gaussian_filter
    verdict, scores = dihedral_verdict(gaussian_filter(truth, 2.0), truth)
    assert verdict == "identity", f"smeared glyph misidentified as {verdict}"
    print(f"{'smeared':15s} identified as identity, "
          f"margin {scores['identity'] - sorted(scores.values())[-2]:.3f}")

    print("\nglyph self-test: all D4 transforms uniquely identified.")


if __name__ == "__main__":
    _self_test()
