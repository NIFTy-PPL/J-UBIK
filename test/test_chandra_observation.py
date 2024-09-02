# test_chandra_observation.py
import nifty8 as ift
import jubik0 as ju
import pytest

@pytest.fixture
def sample_obs_info():
    return {
        'event_file': 'sample_evt2.fits',
        'aspect_sol': 'sample_asol1.fits',
        'bpix_file': 'sample_bpix1.fits',
        'mask_file': 'sample_mask1.fits',
        'data_location': '/path/to/data/'
    }

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

    assert chandra_obs.obsInfo['event_file'] == '/path/to/data/sample_evt2.fits'
    assert chandra_obs.obsInfo['aspect_sol'] == '/path/to/data/sample_asol1.fits'
    assert chandra_obs.obsInfo['bpix_file'] == '/path/to/data/sample_bpix1.fits'
    assert chandra_obs.obsInfo['mask_file'] == '/path/to/data/sample_mask1.fits'
    assert chandra_obs.obsInfo['data_location'] == '/path/to/data/'
    assert chandra_obs.npix_s == npix_s
    assert chandra_obs.npix_e == npix_e
    assert chandra_obs.fov == fov
    assert chandra_obs.elim == elim
    assert chandra_obs.center == center
    assert chandra_obs.energy_ranges == energy_ranges
    assert chandra_obs.chips_off == chips_off