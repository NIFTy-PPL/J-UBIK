def message_obs(obsInfo, verbose=1):
    if verbose > 0:

        print('')
        print('Observation Nr. {:d} Information:'.format(obsInfo['obsID']))
        print('------------------------')
        print('Nominal pointing: RA = {:.4f}, DEC = {:.4f}, ROLL = {:.4f} degrees'.format(obsInfo['ra'], obsInfo['dec'], obsInfo['roll']))
        print('Observation Duration = {:.4f} s'.format(obsInfo['duration']))
        print('')

        print('Considered range in detector coordinates x = {0:.1f} - {1:.1f}, y = {2:.1f} - {3:.1f} pixels'.format(obsInfo['x_min'],\
               obsInfo['x_max'], obsInfo['y_min'], obsInfo['y_max']))
        print('Considered energy range = {0:.1f} to {1:.1f} keV'.format(obsInfo['energy_min'], obsInfo['energy_max']))
        print('------------------------')


def message_binning(obsInfo, verbose=1):
    if verbose > 0:
        print('')
        print('Observation Nr. {:d} Binning:'.format(obsInfo['obsID']))
        print('------------------------')
        print('Total Number of events = ?') # FIXME ?
        print('Number of events considered = {:d}'.format(obsInfo['ntot_binned']))
        print('------------------------')


def message_exposure(obsInfo, verbose=1):
    if verbose > 0:
        print('')
        print('Observation Nr. {:d} Exposure:'.format(obsInfo['obsID']))
        print('------------------------')
        print('Chips online = ', obsInfo['chips_on'])
        print('Chips in the region of interest = ', obsInfo['chips_in'])


def log_file_exists(filename):
    log = f'Output file {filename} already exists and is not regenerated. '\
          'If the observation parameters have changed please'\
           ' delete or rename the current output file.'
    print(log)