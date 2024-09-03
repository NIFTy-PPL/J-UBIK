from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import jubik0 as ju

pmp = pytest.mark.parametrize


@pmp("exposure_filenames", [["test_data/tm1_test_expmap_emin0.2_emax1.0.fits"],
                            ["test_data/tm1_test_expmap_emin0.2_emax1.0.fits",
                             "test_data/tm2_test_expmap_emin0.2_emax1.0.fits"],
                            ])
@pmp("exposure_cut", [None, 12., 100])
def test_build_callable_from_exposure_file(
    exposure_filenames,
    exposure_cut):
    assert ju.build_callable_from_exposure_file(ju.build_exposure_function,
                                                exposure_filenames,
                                                exposure_cut=exposure_cut)


@pmp("psf_filenames", [["test_data/tm1_2dpsf_190219v05.fits",
                        "test_data/tm2_2dpsf_190219v05.fits"]])
@pmp("energies", [["3000"]])
@pmp("pointing_center", [[(1., 1.), (1., 1.)]])
@pmp("domain", [ju.Domain(shape=(128, 128), distances=(0.1, 0.1))])
@pmp("n_patch", [1, 4])
@pmp("margfrac", [0.1])
def test_build_erosita_psf(
    psf_filenames,
    energies,
    pointing_center,
    domain,
    n_patch,
    margfrac
):
    assert ju.build_erosita_psf(psf_filenames,
                                energies,
                                pointing_center,
                                domain,
                                n_patch,
                                margfrac)


@pmp("path_to_caldb", ["test_data/"])
@pmp("tm_ids", [[1,], [1, 2], [3], [1, 2]])
@pmp("e_min, e_max", [
    ([0.2], [2.]),
    ([0.2, 2.], [1.0, 2.5]),
    ([0.2, 1.5], [1.5, 3.0]),
    ([0.3], [0.3])
])
@pmp("caldb_folder_name", ["caldb"])
@pmp("arf_filename_suffix", ["_arf_filter_000101v02.fits"])
@pmp("n_points", [10, 500])
def test_calculate_effective_area(
    path_to_caldb,
    tm_ids,
    e_min,
    e_max,
    caldb_folder_name,
    arf_filename_suffix,
    n_points,
):
    mock_fits_data = {
        'ENERG_LO': np.array([0.1, 1.0, 2.0]),
        'ENERG_HI': np.array([1.0, 2.0, 3.0]),
        'SPECRESP': np.array([100, 200, 300])
    }

    with patch('astropy.io.fits.open') as mock_fits_open:
        mock_fits = MagicMock()
        mock_fits.__enter__.return_value = {'SPECRESP':
                                                MagicMock(data=mock_fits_data)}
        mock_fits_open.return_value = mock_fits

        if tm_ids and e_min and e_max:
            effective_areas = ju.calculate_erosita_effective_area(
                path_to_caldb,
                tm_ids,
                e_min,
                e_max,
                caldb_folder_name=caldb_folder_name,
                arf_filename_suffix=arf_filename_suffix,
                n_points=n_points
            )
            print(effective_areas)

            assert effective_areas.shape == (len(tm_ids), len(e_min))
        # assert np.allclose(effective_areas, expected_values) # TODO

        else:
            # Test for handling of empty or invalid input
            with pytest.raises(ValueError):
                ju.calculate_erosita_effective_area(
                    path_to_caldb,
                    tm_ids,
                    e_min,
                    e_max,
                    caldb_folder_name=caldb_folder_name,
                    arf_filename_suffix=arf_filename_suffix,
                    n_points=n_points
                )


@pmp("s_dim", [20])
@pmp("e_dim", [1])
@pmp("e_min", [[0.]])
@pmp("e_max", [[2.]])
@pmp("fov", [1000])
@pmp("tm_ids", [[1, 2]])
@pmp("exposure_filenames",
     [["test_data/tm1_test_expmap_emin0.2_emax1.0.fits",
       "test_data/tm2_test_expmap_emin0.2_emax1.0.fits", ]])
@pmp("psf_filenames", [["test_data/tm1_2dpsf_190219v05.fits",
                        "test_data/tm2_2dpsf_190219v05.fits"]])
@pmp("psf_energy", [["3000"]])
@pmp("pointing_center", [[(1., 1.), (1., 1.)]])
@pmp("n_patch", [1])
@pmp("margfrac", [0.1])
@pmp("exposure_threshold", [0.1])
@pmp("path_to_caldb", [None, "test_data/"])
@pmp("caldb_folder_name", [None, "caldb"])
@pmp("arf_filename_suffix", [None, "_arf_filter_000101v02.fits"])
@pmp("effective_area_correction", [False, False])
def test_build_erosita_response(
    s_dim,
    e_dim,
    e_min,
    e_max,
    fov,
    tm_ids,
    exposure_filenames,
    psf_filenames,
    psf_energy,
    pointing_center,
    n_patch,
    margfrac,
    exposure_threshold,
    path_to_caldb,
    caldb_folder_name,
    arf_filename_suffix,
    effective_area_correction,
):

    response_dict = ju.build_erosita_response(
        s_dim,
        e_dim,
        e_min,
        e_max,
        fov,
        tm_ids,
        exposure_filenames,
        psf_filenames,
        psf_energy,
        pointing_center,
        n_patch,
        margfrac,
        exposure_threshold,
        path_to_caldb,
        caldb_folder_name,
        arf_filename_suffix,
        effective_area_correction,
    )
    assert isinstance(response_dict, dict)
    assert 'pix_area' in response_dict and isinstance(response_dict['pix_area'],
                                                      float)
    assert 'psf' in response_dict and callable(response_dict['psf'])
    assert 'exposure' in response_dict and callable(response_dict['exposure'])
    assert 'mask' in response_dict and callable(response_dict['mask'])
    assert 'R' in response_dict and callable(response_dict['R'])
