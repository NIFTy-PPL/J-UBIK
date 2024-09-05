def message_obs(obsInfo, verbose=1):
    """
    Print observation information for Chandra.

    This function prints detailed information about a Chandra observation,
    including nominal pointing, observation duration, detector coordinate
    range, and energy range.

    Parameters
    ----------
    obsInfo : dict
        ChandraObservation dictionary containing observation information.
         Expected keys include:
        - 'obsID': Observation ID.
        - 'ra': Right Ascension of the nominal pointing (in degrees).
        - 'dec': Declination of the nominal pointing (in degrees).
        - 'roll': Roll angle of the nominal pointing (in degrees).
        - 'duration': Duration of the observation (in seconds).
        - 'x_min': Minimum x-coordinate in detector coordinates (in pixels).
        - 'x_max': Maximum x-coordinate in detector coordinates (in pixels).
        - 'y_min': Minimum y-coordinate in detector coordinates (in pixels).
        - 'y_max': Maximum y-coordinate in detector coordinates (in pixels).
        - 'energy_min': Minimum energy considered (in keV).
        - 'energy_max': Maximum energy considered (in keV).
    verbose : int, optional
        Verbosity level. If greater than 0, the information is printed.
        Default is 1.

    Returns
    -------
    None
    """
    if verbose > 0:
        print('')
        print('Observation Nr. {:d} Information:'.format(obsInfo['obsID']))
        print('------------------------')
        print('Nominal pointing: RA = {:.4f}, DEC = {:.4f}, ROLL = {:.4f} \
        degrees'.format(obsInfo['ra'], obsInfo['dec'], obsInfo['roll']))
        print('Observation Duration = {:.4f} s'.format(obsInfo['duration']))
        print('')
        print('Considered range in detector coordinates x = {0:.1f} - {1:.1f},\
         y = {2:.1f} - {3:.1f} pixels'.format(obsInfo['x_min'],
                                              obsInfo['x_max'],
                                              obsInfo['y_min'],
                                              obsInfo['y_max']))
        print('Considered energy range = {0:.1f} to {1:.1f} keV'.format(
            obsInfo['energy_min'], obsInfo['energy_max']))
        print('------------------------')
    if verbose > 0:

        print('')
        print('Observation Nr. {:d} Information:'.format(obsInfo['obsID']))
        print('------------------------')
        print('Nominal pointing: RA = {:.4f}, DEC = {:.4f}, ROLL = {:.4f} \
        degrees'.format(obsInfo['ra'], obsInfo['dec'], obsInfo['roll']))
        print('Observation Duration = {:.4f} s'.format(obsInfo['duration']))
        print('')

        print('Considered range in detector coordinates x = {0:.1f} - {1:.1f},\
        y = {2:.1f} - {3:.1f} pixels'.format(obsInfo['x_min'],\
               obsInfo['x_max'], obsInfo['y_min'], obsInfo['y_max']))
        print('Considered energy range = {0:.1f} to {1:.1f} keV'.format(
            obsInfo['energy_min'], obsInfo['energy_max']))
        print('------------------------')


def message_binning(obsInfo, verbose=1):
    """
    Print binning information for a Chandra observation.

    This function prints information about the binning of events in an
    observation, including the total number of events and the number of
    events considered.

    Parameters
    ----------
    obsInfo : dict
        A dictionary containing observation information. Expected keys include:
        - 'obsID': Observation ID.
        - 'ntot_binned': Number of events considered in the binning.
    verbose : int, optional
        Verbosity level. If greater than 0, the information is printed.
        Default is 1.

    Returns
    -------
    None
    """
    if verbose > 0:
        print('')
        print('Observation Nr. {:d} Binning:'.format(obsInfo['obsID']))
        print('------------------------')
        print('Number of events considered = {:d}'.format(
            obsInfo['ntot_binned']))
        print('------------------------')


def message_exposure(obsInfo, verbose=1):
    """
    Print exposure information for a Chandra observation.

    This function prints information about the exposure of an observation,
    including the chips that are online and those in the region of interest.

    Parameters
    ----------
    obsInfo : dict
        A dictionary containing observation information. Expected keys include:
        - 'obsID': Observation ID.
        - 'chips_on': List of chips that are online.
        - 'chips_in': List of chips in the region of interest.
    verbose : int, optional
        Verbosity level. If greater than 0, the information is printed.
         Default is 1.

    Returns
    -------
    None
    """
    if verbose > 0:
        print('')
        print('Observation Nr. {:d} Exposure:'.format(obsInfo['obsID']))
        print('------------------------')
        print('Chips online = ', obsInfo['chips_on'])
        print('Chips in the region of interest = ', obsInfo['chips_in'])


def log_file_exists(filename):
    """
    Log a message indicating that a file already exists.

    This function prints a message indicating that the specified output file
    already exists and will not be regenerated. It advises the user to delete
    or rename the current file if the observation parameters have changed.

    Parameters
    ----------
    filename : str
        The name of the file that already exists.

    Returns
    -------
    None
    """
    log = f'Output file {filename} already exists and is not regenerated. '\
          'If the observation parameters have changed please'\
           ' delete or rename the current output file.'
    print(log)