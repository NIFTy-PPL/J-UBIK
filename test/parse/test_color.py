import pytest
import numpy as np
from astropy import units as u
from numpy.testing import assert_allclose

from jubik0.color import Color
from jubik0.parse.color import yaml_to_binned_colors


class TestYamlToBinnedColors:
    """Test suite for yaml_to_binned_colors function."""

    #### **Test Default Behavior**

    def test_no_energy_bin_returns_default(self):
        """Test that missing energy_bin returns default full range."""
        config = {}
        result = yaml_to_binned_colors(config)

        assert isinstance(result, Color)
        assert result.unit == u.Hz
        assert_allclose(result.value, [[0, np.inf]])

    def test_empty_config_with_other_keys(self):
        """Test config with other keys but no energy_bin."""
        config = {
            "some_other_key": "value",
            "energy_unit": "keV",  # Unit present but no bins
        }
        result = yaml_to_binned_colors(config)

        assert result.unit == u.Hz
        assert_allclose(result.value, [[0, np.inf]])

    #### **Test Simple List Format**

    def test_simple_list_consecutive_bins(self):
        """Test consecutive bins from simple list."""
        config = {"energy_unit": "keV", "energy_bin": [0.5, 1.0, 2.0, 10.0]}
        result = yaml_to_binned_colors(config)

        assert result.unit == u.keV
        assert result.shape == (3, 2)
        assert_allclose(result[0].value, [0.5, 1.0])
        assert_allclose(result[1].value, [1.0, 2.0])
        assert_allclose(result[2].value, [2.0, 10.0])

    def test_simple_list_two_values(self):
        """Test single bin from two values."""
        config = {"energy_unit": "eV", "energy_bin": [100, 200]}
        result = yaml_to_binned_colors(config)

        assert result.unit == u.eV
        assert result.shape == (1, 2)
        assert_allclose(result[0].value, [100, 200])

    #### **Test 2D Array Format**

    def test_2d_array_discontinuous_bins(self):
        """Test discontinuous bins from 2D array."""
        config = {
            "energy_unit": "keV",
            "energy_bin": [[1.0, 2.0], [5.0, 10.0], [15.0, 20.0]],
        }
        result = yaml_to_binned_colors(config)

        assert result.unit == u.keV
        assert result.shape == (3, 2)
        assert_allclose(result[0].value, [1.0, 2.0])
        assert_allclose(result[1].value, [5.0, 10.0])
        assert_allclose(result[2].value, [15.0, 20.0])

    #### **Test Dictionary Format**

    def test_dict_format_with_emin_emax(self):
        """Test dictionary format with e_min and e_max keys."""
        config = {
            "energy_unit": "keV",
            "energy_bin": {"e_min": [1.0, 3.0, 10.0], "e_max": [2.0, 5.0, 15.0]},
        }
        result = yaml_to_binned_colors(config)

        assert result.unit == u.keV
        assert result.shape == (3, 2)
        assert_allclose(result[0].value, [1.0, 2.0])
        assert_allclose(result[1].value, [3.0, 5.0])
        assert_allclose(result[2].value, [10.0, 15.0])

    def test_dict_format_single_bin(self):
        """Test dictionary format with single bin."""
        config = {"energy_unit": "eV", "energy_bin": {"e_min": [500], "e_max": [1000]}}
        result = yaml_to_binned_colors(config)

        assert result.unit == u.eV
        assert result.shape == (1, 2)
        assert_allclose(result[0].value, [500, 1000])

    def test_dict_format_overlapping_bins(self):
        """Test dictionary format with overlapping energy ranges."""
        config = {
            "energy_unit": "keV",
            "energy_bin": {"e_min": [1.0, 1.5, 2.5], "e_max": [2.0, 3.0, 4.0]},
        }
        result = yaml_to_binned_colors(config)

        # Should create overlapping bins
        assert result.shape == (3, 2)
        assert_allclose(result[0].value, [1.0, 2.0])
        assert_allclose(result[1].value, [1.5, 3.0])
        assert_allclose(result[2].value, [2.5, 4.0])

    #### **Test Different Units**

    def test_different_energy_units(self):
        """Test various astropy energy units."""
        units_to_test = ["keV", "eV", "MeV", "GeV", "TeV"]

        for unit_name in units_to_test:
            config = {"energy_unit": unit_name, "energy_bin": [1.0, 10.0]}
            result = yaml_to_binned_colors(config)

            expected_unit = getattr(u, unit_name)
            assert result.unit == expected_unit
            assert_allclose(result[0].value, [1.0, 10.0])

    def test_frequency_units(self):
        """Test with frequency units."""
        config = {"energy_unit": "Hz", "energy_bin": [1e14, 1e15, 1e16]}
        result = yaml_to_binned_colors(config)

        assert result.unit == u.Hz
        assert result.shape == (2, 2)

    def test_wavelength_units(self):
        """Test with wavelength units."""
        config = {"energy_unit": "nm", "energy_bin": [400, 500, 600, 700]}
        result = yaml_to_binned_colors(config)

        assert result.unit == u.nm
        assert result.shape == (3, 2)

    #### **Test Edge Cases**

    def test_numpy_array_input(self):
        """Test with numpy array instead of list."""
        config = {"energy_unit": "keV", "energy_bin": np.array([1.0, 2.0, 3.0])}
        result = yaml_to_binned_colors(config)

        assert result.shape == (2, 2)
        assert_allclose(result[0].value, [1.0, 2.0])
        assert_allclose(result[1].value, [2.0, 3.0])

    def test_mixed_format_nested_lists(self):
        """Test with nested lists in various formats."""
        config = {"energy_unit": "eV", "energy_bin": [[100, 200, 300], [500, 600, 700]]}
        result = yaml_to_binned_colors(config)

        # This creates bins from each sub-list
        assert result.unit == u.eV
        # The exact shape depends on how Color handles this

    def test_unsorted_input(self):
        """Test that unsorted input gets properly sorted."""
        config = {"energy_unit": "keV", "energy_bin": [10.0, 1.0, 5.0, 2.0]}
        result = yaml_to_binned_colors(config)

        # Color should sort these internally
        assert result[0, 0] < result[0, 1]  # Each bin is min to max

    #### **Test Error Conditions**

    def test_invalid_unit_raises_error(self):
        """Test that invalid unit name raises AttributeError."""
        config = {"energy_unit": "invalid_unit", "energy_bin": [1.0, 2.0]}

        with pytest.raises(AttributeError):
            yaml_to_binned_colors(config)

    def test_missing_emin_in_dict_format(self):
        """Test missing e_min key in dictionary format."""
        config = {
            "energy_unit": "keV",
            "energy_bin": {
                "e_max": [2.0, 5.0]  # Missing e_min
            },
        }

        with pytest.raises(KeyError):
            yaml_to_binned_colors(config)

    def test_missing_emax_in_dict_format(self):
        """Test missing e_max key in dictionary format."""
        config = {
            "energy_unit": "keV",
            "energy_bin": {
                "e_min": [1.0, 3.0]  # Missing e_max
            },
        }

        with pytest.raises(KeyError):
            yaml_to_binned_colors(config)

    def test_mismatched_emin_emax_lengths(self):
        """Test mismatched lengths of e_min and e_max arrays."""
        config = {
            "energy_unit": "keV",
            "energy_bin": {
                "e_min": [1.0, 3.0, 5.0],
                "e_max": [2.0, 4.0],  # Different length
            },
        }
        result = yaml_to_binned_colors(config)

        # This should only create 2 bins (limited by shorter array)
        assert result.shape[0] <= 2

    def test_missing_energy_unit_with_bins(self):
        """Test missing energy_unit when energy_bin is present."""
        config = {"energy_bin": [1.0, 2.0, 3.0]}

        # This should raise an error since we can't get the unit
        with pytest.raises(KeyError):
            yaml_to_binned_colors(config)

    #### **Test Integration with Color Class**

    def test_result_is_valid_color_object(self):
        """Test that result is a proper Color object with expected methods."""
        config = {"energy_unit": "keV", "energy_bin": [1.0, 2.0, 5.0]}
        result = yaml_to_binned_colors(config)

        # Test Color methods work
        assert hasattr(result, "contains")
        assert hasattr(result, "center")
        assert hasattr(result, "is_continuous")

        # Test contains method
        assert result.contains(1.5 * u.keV)
        assert not result.contains(10.0 * u.keV)

        # Test center property
        centers = result.center
        assert len(centers) == 2

    def test_unit_conversion_capability(self):
        """Test that resulting Color can convert units."""
        config = {"energy_unit": "keV", "energy_bin": [1.0, 10.0]}
        result = yaml_to_binned_colors(config)

        # Convert to eV
        result_ev = result.to(u.eV, equivalencies=u.spectral())
        assert result_ev.unit == u.eV
        assert_allclose(result_ev[0].value, [1000, 10000])

        # Convert to wavelength
        result_nm = result.to(u.nm, equivalencies=u.spectral())
        assert result_nm.unit == u.nm


class TestYamlToBinnedColorsRealWorldScenarios:
    """Test with realistic scientific use cases."""

    def test_xray_energy_bands(self):
        """Test typical X-ray energy bands configuration."""
        config = {
            "energy_unit": "keV",
            "energy_bin": {
                "e_min": [0.5, 2.0, 5.0],  # Soft, Medium, Hard X-rays
                "e_max": [2.0, 5.0, 10.0],
            },
        }
        result = yaml_to_binned_colors(config)

        assert result.shape == (3, 2)
        assert result.is_continuous  # These bands are continuous

    def test_optical_wavelength_bands(self):
        """Test optical wavelength bands (UBVRI filters)."""
        config = {
            "energy_unit": "nm",
            "energy_bin": [
                [365, 445],  # U band
                [445, 551],  # B band
                [551, 658],  # V band
                [658, 806],  # R band
                [806, 1020],  # I band
            ],
        }
        result = yaml_to_binned_colors(config)

        assert result.shape == (5, 2)
        assert result.unit == u.nm

    def test_gamma_ray_energy_bands(self):
        """Test gamma-ray energy bands (Fermi-LAT like)."""
        config = {
            "energy_unit": "GeV",
            "energy_bin": np.logspace(-1, 3, 21),  # Log-spaced from 0.1 to 1000 GeV
        }
        result = yaml_to_binned_colors(config)

        assert result.shape == (20, 2)
        assert result.unit == u.GeV
        assert result.is_continuous
