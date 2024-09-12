# test_chandra_observation.py
from os.path import join

import pytest

import jubik0 as ju


@pytest.fixture
def sample_obs_info():
    return {
        'obsID': 00000,
        'event_file': 'primary/structure_only_evt.fits',
        'aspect_sol': 'primary/structure_only_asol.fits',
        'bpix_file': 'primary/structure_only_bpix.fits',
        'mask_file': 'secondary/structure_only_msk.fits',
        'data_location': 'chandra_test_data/'
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

    assert chandra_obs.obsInfo['event_file'] == join('chandra_test_data/','primary/structure_only_evt.fits')
    assert chandra_obs.obsInfo['aspect_sol'] == join('chandra_test_data/','primary/structure_only_asol.fits')
    assert chandra_obs.obsInfo['bpix_file'] == join('chandra_test_data/','primary/structure_only_bpix.fits')
    assert chandra_obs.obsInfo['mask_file'] == join('chandra_test_data/','secondary/structure_only_msk.fits')
    assert chandra_obs.obsInfo['data_location'] == 'chandra_test_data/'
    assert chandra_obs.obsInfo['obsID'] == 00000
    assert chandra_obs.obsInfo['npix_s'] == npix_s
    assert chandra_obs.obsInfo['npix_e'] == npix_e
    assert chandra_obs.obsInfo['fov'] == fov
    assert chandra_obs.obsInfo['energy_min'] == elim[0]
    assert chandra_obs.obsInfo['energy_max'] == elim[1]
    assert chandra_obs.obsInfo['energy_ranges'] == energy_ranges
    assert chandra_obs.obsInfo['chips_off'] == chips_off

def test_get_data(sample_obs_info):
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

    data_array = chandra_obs.get_data('chandra_test_data/generated_data.fits')
    assert data_array.shape == (npix_s, npix_s, npix_e) 