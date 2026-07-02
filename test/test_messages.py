from jubik.messages import (
    log_file_exists,
    message_binning,
    message_exposure,
    message_obs,
)


def _sample_obs_info():
    return {
        "obsID": 42,
        "ra": 10.0,
        "dec": 20.0,
        "roll": 30.0,
        "duration": 1000.0,
        "x_min": 0.0,
        "x_max": 10.0,
        "y_min": 1.0,
        "y_max": 11.0,
        "energy_min": 0.5,
        "energy_max": 7.0,
        "ntot_binned": 123,
        "chips_on": [0, 1, 2],
        "chips_in": [1, 2],
    }


def test_message_obs_prints_when_verbose(capsys):
    message_obs(_sample_obs_info(), verbose=1)
    out = capsys.readouterr().out

    assert "Observation Nr. 42 Information" in out
    assert "Nominal pointing:" in out
    assert "Considered energy range = 0.5 to 7.0 keV" in out


def test_message_obs_silent_when_not_verbose(capsys):
    message_obs(_sample_obs_info(), verbose=0)
    assert capsys.readouterr().out == ""


def test_message_binning_prints_when_verbose(capsys):
    message_binning(_sample_obs_info(), verbose=1)
    out = capsys.readouterr().out

    assert "Observation Nr. 42 Binning" in out
    assert "Number of events considered = 123" in out


def test_message_binning_silent_when_not_verbose(capsys):
    message_binning(_sample_obs_info(), verbose=0)
    assert capsys.readouterr().out == ""


def test_message_exposure_prints_when_verbose(capsys):
    message_exposure(_sample_obs_info(), verbose=1)
    out = capsys.readouterr().out

    assert "Observation Nr. 42 Exposure" in out
    assert "Chips online =  [0, 1, 2]" in out
    assert "Chips in the region of interest =  [1, 2]" in out


def test_message_exposure_silent_when_not_verbose(capsys):
    message_exposure(_sample_obs_info(), verbose=0)
    assert capsys.readouterr().out == ""


def test_log_file_exists_prints_message(capsys):
    log_file_exists("output.fits")
    out = capsys.readouterr().out

    assert "Output file output.fits already exists and is not regenerated." in out
