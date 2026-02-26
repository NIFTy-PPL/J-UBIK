# test_chandra_observation.py
from collections import Counter

import numpy as np
import pytest
from astropy.io import fits

import jubik as ju

ciao_installed = True

try:
    import ciao_contrib
except ImportError:
    ciao_installed = False


@pytest.fixture
def sample_obs_info(request):
    return {
        "obsID": 00000,
        "event_file": "primary/structure_only_evt.fits",
        "aspect_sol": "primary/structure_only_asol.fits",
        "bpix_file": "primary/structure_only_bpix.fits",
        "mask_file": "secondary/structure_only_msk.fits",
        "data_location": f"{request.path.parent}/chandra_test_data/",
    }


@pytest.mark.skipif(not ciao_installed, reason="CIAO is not installed")
def test_initialization(sample_obs_info):
    npix_s = 1024
    npix_e = 100
    fov = 30.0
    elim = (0.5, 7.0)
    center = (10.684, 41.269)
    energy_ranges = None
    chips_off = (5, 7)

    chandra_obs = ju.ChandraObservationInformation(
        obsInfo=sample_obs_info,
        npix_s=npix_s,
        npix_e=npix_e,
        fov=fov,
        elim=elim,
        center=center,
        energy_ranges=energy_ranges,
        chips_off=chips_off
    )

    assert chandra_obs.obsInfo["event_file"].endswith("primary/structure_only_evt.fits")
    assert chandra_obs.obsInfo["aspect_sol"].endswith(
        "primary/structure_only_asol.fits"
    )
    assert chandra_obs.obsInfo["bpix_file"].endswith("primary/structure_only_bpix.fits")
    assert chandra_obs.obsInfo["mask_file"].endswith(
        "secondary/structure_only_msk.fits"
    )
    assert chandra_obs.obsInfo["data_location"].endswith("chandra_test_data/")
    assert chandra_obs.obsInfo['obsID'] == 00000
    assert chandra_obs.obsInfo['npix_s'] == npix_s
    assert chandra_obs.obsInfo['npix_e'] == npix_e
    assert chandra_obs.obsInfo['fov'] == fov
    assert chandra_obs.obsInfo['energy_min'] == elim[0]
    assert chandra_obs.obsInfo['energy_max'] == elim[1]
    assert chandra_obs.obsInfo['energy_ranges'] == energy_ranges
    assert chandra_obs.obsInfo['chips_off'] == chips_off


@pytest.mark.skipif(not ciao_installed, reason="CIAO is not installed")
def test_get_data(sample_obs_info, request):
    npix_s = 1024
    npix_e = 100
    fov = 30.0
    elim = (0.5, 7.0)
    center = (10.684, 41.269)
    energy_ranges = None
    chips_off = (5, 7)

    chandra_obs = ju.ChandraObservationInformation(
        obsInfo=sample_obs_info,
        npix_s=npix_s,
        npix_e=npix_e,
        fov=fov,
        elim=elim,
        center=center,
        energy_ranges=energy_ranges,
        chips_off=chips_off
    )

    outfile = f"{request.path.parent}/chandra_test_data/generated_data.fits"
    data_array = chandra_obs.get_data(outfile)
    assert data_array.shape == (npix_s, npix_s, npix_e)
    assert np.issubdtype(data_array.dtype, np.integer)
    assert np.isfinite(data_array).all()
    assert (data_array >= 0).all()
    assert data_array.sum() > 0
    assert chandra_obs.obsInfo["ntot_binned"] == data_array.sum()

    # Reconstruct the expected histogram from the filtered event list written by
    # CIAO
    with fits.open(outfile) as dat_filtered:
        evts = dat_filtered["EVENTS"].data

    x_edges = np.linspace(chandra_obs.obsInfo["x_min"], chandra_obs.obsInfo["x_max"], npix_s + 1)
    y_edges = np.linspace(chandra_obs.obsInfo["y_min"], chandra_obs.obsInfo["y_max"], npix_s + 1)
    e_edges = np.linspace(np.log(chandra_obs.obsInfo["energy_min"]),
                          np.log(chandra_obs.obsInfo["energy_max"]),
                          npix_e + 1)

    counts = Counter()
    for x, y, energy in zip(evts["x"], evts["y"], evts["energy"]):
        log_energy = np.log(1.e-3 * energy)

        ix = np.searchsorted(x_edges, x, side="right") - 1
        iy = np.searchsorted(y_edges, y, side="right") - 1
        ie = np.searchsorted(e_edges, log_energy, side="right") - 1

        if ix == npix_s and x == x_edges[-1]:
            ix = npix_s - 1
        if iy == npix_s and y == y_edges[-1]:
            iy = npix_s - 1
        if ie == npix_e and log_energy == e_edges[-1]:
            ie = npix_e - 1

        if 0 <= ix < npix_s and 0 <= iy < npix_s and 0 <= ie < npix_e:
            counts[(iy, ix, ie)] += 1

    assert data_array.sum() == sum(counts.values())
    assert data_array.sum() <= len(evts)
    for bin_idx, count in counts.items():
        assert data_array[bin_idx] == count
