"""p8 — the visibility-seam witness: ONE crossing, external data, no theory.

WHAT THIS MEASURES (no assertions — read the numbers yourself)
    Forward-models the CASA-minted truth sky (roundtrip_radio_truth.fits)
    through the CURRENT resolve gridder backend under three hypotheses
    about the sky layout / conjugation, and correlates each directly with
    the CASA visibilities (roundtrip_radio_obs.npz).  CASA is the only
    authority here; nothing is compared against an analytic formula
    written in this repo.

        corr(B(truth.T),        data)   sky in [RA, Dec] gridder layout
        corr(B(truth),          data)   sky in [Dec, RA] canonical layout
        corr(conj(B(truth.T)),  data)   [RA, Dec] + a conjugation

    One number near 1 identifies the composition the data actually
    demand.  On the reverted (pre-canonical) code the expected winner is
    the first row — the validated-production composition.  After the
    canonical adapter (pure transpose) lands, running the adapter
    version of this witness must put the canonical row near 1.

    This is the one-crossing test that the image-domain roundtrip (p7)
    is mathematically blind to: a conjugation cancels against the
    bilinear adjoint in a forward+adjoint round trip, but cannot hide
    in a single forward crossing against external data.

RUN (from the repo root)
    uv run python probes/p8_vis_seam_witness.py
"""

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from astropy.io import fits

from jubik.instruments.resolve.data.observation import Observation
from jubik.instruments.resolve.response import interferometry_response_ducc

GOLDEN_DIR = Path(__file__).parent / "golden"


def main() -> None:
    obs = Observation.load(str(GOLDEN_DIR / "roundtrip_radio_obs.npz"))
    with fits.open(GOLDEN_DIR / "roundtrip_radio_truth.fits") as hdul:
        truth = np.squeeze(hdul[0].data).astype(np.float64)  # [Dec, RA] canonical
        dpix_rad = abs(hdul[0].header["CDELT1"]) * np.pi / 180.0

    stub = SimpleNamespace(uvw=np.asarray(obs.uvw), freq=np.asarray(obs.freq))
    B = interferometry_response_ducc(
        stub, npix_x=truth.shape[1], npix_y=truth.shape[0],
        pixsize_x=dpix_rad, pixsize_y=dpix_rad,
        do_wgridding=False, epsilon=1e-5, nthreads=1, verbosity=0,
    )
    d = np.asarray(obs.vis_val[0]).ravel()

    def corr(v: np.ndarray) -> float:
        v = np.asarray(v).ravel()
        return float(np.abs(np.vdot(v, d)) / (np.linalg.norm(v) * np.linalg.norm(d)))

    rows = [
        ("B(truth.T)        [RA,Dec] layout, no conj ", corr(B(truth.T))),
        ("B(truth)           [Dec,RA] canonical, no conj", corr(B(truth))),
        ("conj(B(truth.T))   [RA,Dec] + conjugation    ", corr(np.conj(B(truth.T)))),
    ]
    print("correlation of forward model against the CASA visibilities:\n")
    for label, value in rows:
        print(f"    {label}   {value:.4f}")
    print("\n(the composition the data demand is the row near 1; "
          "everything else follows from that number, not from any claim)")


if __name__ == "__main__":
    main()
