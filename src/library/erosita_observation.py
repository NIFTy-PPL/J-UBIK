# FIXME: add copyright
import os
import subprocess
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import colors
import numpy as np

from .utils import _check_type


class ErositaObservation:
    """
    Base class to retrieve and process eROSITA data.


    # Input datasets: eSASS event files or images with a full set of eSASS standard file
    extensions. FIXME

    """

    def __init__(self, input_filename, output_filename, working_directory):
        # TODO: Add parameter checks
        self.working_directory = os.path.abspath(working_directory)
        self.input = input_filename
        self.output = output_filename
        self.image = " erosita/esass:latest "

        # TODO: Add all fits file fields

        self._mounted_dir = "/home/idies/mnt_dir/"
        self._base_command = "docker run --platform linux/amd64 --volume " + \
                             self.working_directory + ":" + \
                             self._mounted_dir + \
                             self.image + "/bin/bash -c 'source ~/.bashrc && "

    def get_data(self, **kwargs):
        """
        Allows to extract and manipulate data from eROSITA event files through the eSASS 'evtool'
        command.

        Parameters
        ----------

        **kwargs : keyword arguments to be passed to _get_evtool_flags
        """
        print("Collecting data from {}.".format(self.input))
        input_files = self._parse_stringlists(self.input, additional_path=self._mounted_dir)
        output_file = self._mounted_dir + self.output

        flags = self._get_evtool_flags(**kwargs)
        command = self._base_command + 'evtool ' + input_files + " " + output_file + flags + "'"

        self._run_task(command)
        print("The processed dataset has been saved as {}.".format(
            os.path.join(self.working_directory, self.output)))
        return fits.open(os.path.join(self.working_directory, self.output))

    def get_exposure_maps(self, template_image, emin, emax, badpix_correction=True, **kwargs):
        """
        Computes exposure maps for eROSITA event files through the eSASS 'expmap' command.

        Parameters
        ----------
        template_image: str
        Path to the output exposure maps will be binned as specified in the WCS keywords of the
        template image.
        emin: float
        emax: float
        badpix_correction: bool (default: True)
        Loads the corrected eROSITA detmaps. To build the bad-pixel corrected maps,
        use the auxiliary function create_erosita_badpix_to_detmaps.
        If withinputmaps=YES: input exposure maps
        """
        # TODO: parameter checks
        input_files = self._parse_stringlists(self.input, additional_path=self._mounted_dir)
        template_image = os.path.join(self._mounted_dir, template_image)
        flags = self._get_exmap_flags(self._mounted_dir, template_image, emin, emax, **kwargs)
        command = self._base_command + 'expmap ' + input_files + flags + "'"

        caldb_loc_base = '/home/idies/caldb/data/erosita/tm{}/bcf/'
        detmap_file_base = 'tm{}_detmap_100602v02.fits'

        if badpix_correction:
            command = self._base_command
            for i in range(7):
                caldb_loc = caldb_loc_base.format(i+1)
                detmap_file = detmap_file_base.format(i+1)
                update_detmap_command = f' mv {caldb_loc}{detmap_file} ' \
                                f'{caldb_loc}tm{i+1}_detmap_100602v02_old.fits && ' \
                      f'cp {self._mounted_dir}new_detmaps/{detmap_file} {caldb_loc} &&'
                command += update_detmap_command
            command += ' expmap ' + input_files + flags + "'"

        self._run_task(command)

    def load_fits_data(self, filename, verbose=True):
        if verbose:
            print("Loading ouput data stored in {}.".format(filename))
        return fits.open(os.path.join(self.working_directory, filename))

    def get_pointing_coordinates_stats(self, module, input_filename=None):
        if not isinstance(module, int):
            raise TypeError('Telescope module must be of type int (1-7).')
        module = 'CORRATT' + str(module)

        if input_filename is None:
            input_filename = self.input
        print('Loading pointing coordinates for TM{} module from {}.'.format(module[-1],
                                                                             input_filename))

        try:
            data = self.load_fits_data(input_filename, verbose=False)[module].data
            
        except ValueError as err:
            raise ValueError(
            f"""
            Input filename does not contain pointing information.
            
            {err}
            """
            )

        # Convert pointing information to arcseconds
        conv = 3600
        ra = conv * data['RA']
        dec = conv * data['DEC']
        roll = conv * data['ROLL']

        # Return pointing statistics
        stats = {'RA': (ra.mean(), ra.std()), 'DEC': (dec.mean(), dec.std()), 'ROLL': (
            roll.mean(), roll.std())}

        return stats

    def plot_fits_data(self, filename, image_name, slice=None, lognorm=True, linthresh=10e-1,
                       show=False, dpi=None, **kwargs):
        im = self.load_fits_data(filename)[0].data
        if slice is not None:
            slice = tuple(slice)
            im = im[slice[2]:slice[3], slice[0]:slice[1]]
        output = os.path.join(self.working_directory, image_name)
        norm = None
        if lognorm:
            norm = colors.SymLogNorm(linthresh=linthresh)
        plt.imshow(im, origin='lower', norm=norm, **kwargs)
        plt.colorbar()
        if show:
            plt.show()
        plt.savefig(output, dpi=dpi)
        plt.close()
        print(filename + " data image saved as {}.".format(output))

    @staticmethod
    def _run_task(command):
        proc = subprocess.Popen([command], shell=True)
        proc.wait()
        (stdout, stderr) = proc.communicate()
        if proc.returncode != 0:
            raise FileNotFoundError("Docker Error")
        else:
            print("eSASS task COMPLETE.")

    @staticmethod
    def _get_evtool_flags(clobber=True, events=True, image=False, size=None,
                          rebin=None, center_position=None, region=None,
                          gti=None, flag=None, flag_invert=None, pattern=None,
                          telid=None, emin=None, emax=None, rawxy=None,
                          rawxy_telid=None, rawxy_invert=False, memset=None,
                          overlap=None, skyfield=None):

        input_params = {'clobber': bool, 'events': bool, 'image': bool, 'size': int,
                        'rebin': int, 'center_position': tuple, 'region': str,
                        'gti': str, 'flag': str, 'flag_invert': bool, 'pattern': int,
                        'telid': int, 'emin': float, 'emax': float, 'rawxy': str,
                        'rawxy_telid': int, 'rawxy_invert': bool, 'memset': int,
                        'overlap': float, 'skyfield': str}

        # Implements type checking
        for key, val in input_params.items():
            _check_type(eval(key), val, name=key)

        """
        Returns appropriate evtool command flags.

        Parameters
        ----------

        clobber: bool
        events: bool
        image: bool
        size: int
        rebin: int
        center_position
        region: str
        gti: str
        flag: str
        flag_invert: bool
        pattern: int
        telid: str
        emin: float
        emax: float
        rawxy: str
        rawxy_telid: int
        rawxy_invert: bool
        memset: int
        overlap: float
        skyfield: str
        """

        flags = ""
        flags += "" if clobber else " clobber=no"
        flags += "" if events else " events=no"
        flags += " image=yes" if image else " image=no"
        flags += " size={}".format(size) if size is not None else ""
        flags += " rebin={}".format(rebin) if rebin is not None else ""
        flags += " center_position={}".format(
            center_position) if center_position is not None else ""
        flags += " region={}".format(region) if region is not None else ""
        flags += " gti={}".format(gti) if gti is not None else ""
        flags += " flag={}".format(flag) if flag is not None else ""
        flags += " flag_invert=yes" if flag_invert else ""
        flags += " pattern={}".format(pattern) if pattern is not None else ""
        flags += " telid={}".format(telid) if telid is not None else ""
        flags += " emin={}".format(emin) if emin is not None else ""
        flags += " emax={}".format(emax) if emax is not None else ""
        flags += " rawxy={}".format(rawxy) if rawxy is not None else ""
        flags += " rawxy_telid={}".format(rawxy_telid) if rawxy_telid is not None else ""
        flags += " rawxy_invert=yes" if rawxy_invert else ""
        flags += " memset={}".format(memset) if memset is not None else ""
        flags += " overlap={}".format(overlap) if overlap is not None else ""
        flags += " skyfield={}".format(skyfield) if skyfield is not None else ""

        return flags

    @staticmethod
    def _get_exmap_flags(mounted_dir, templateimage, emin, emax,
                         withsinglemaps=False, withmergedmaps=False, singlemaps=None,
                         mergedmaps=None, gtitype='GTI', withvignetting=True,
                         withdetmaps=True, withweights=True, withfilebadpix=True,
                         withcalbadpix=True, withinputmaps=False):

        input_params = {'mounted_dir': str, 'templateimage': str, 'emin': float, 'emax': float,
                        'withsinglemaps': bool, 'withmergedmaps': bool, 'singlemaps': list,
                        'mergedmaps': str, 'gtitype': str, 'withvignetting': bool,
                        'withdetmaps': bool, 'withweights': bool, 'withfilebadpix': bool,
                        'withcalbadpix': bool, 'withinputmaps': bool}

        # Implements type checking
        for key, val in input_params.items():
            _check_type(eval(key), val, name=key)

        if singlemaps is not None:
            singlemaps = list(map(lambda x: os.path.join(mounted_dir, x), singlemaps))
            singlemaps_str = '"' + " ".join(singlemaps) + '"'

        flags = " "
        flags += templateimage if templateimage is not None else print(
            "template image cannot be None.")  # FIXME Add exit somehow
        flags += " emin={}".format(emin) if emin is not None else print("emin cannot be None.")
        flags += " emax={}".format(emax) if emax is not None else print("emax cannot be None.")
        flags += " withsinglemaps=yes" if withsinglemaps else ""
        flags += "" if withmergedmaps else " withmergedmaps=no"
        flags += " singlemaps={}".format(singlemaps_str) if singlemaps is not None else ""
        flags += " mergedmaps={}".format(
            os.path.join(mounted_dir, mergedmaps)) if mergedmaps is not None else ""
        flags += " gtitype={}".format(gtitype) if gtitype != "GTI" else ""
        flags += "" if withvignetting else " withvignetting=no"
        flags += " withdetmaps=yes" if withdetmaps else ""
        flags += "" if withweights else " withweights=no"
        flags += "" if withfilebadpix else " withfilebadpix=no"
        flags += "" if withcalbadpix else " withcalbadpix=no"
        flags += " withinputmaps=yes" if withinputmaps else ""

        return flags

    @staticmethod
    def _parse_stringlists(stringlist, additional_path: str = ""):
        if isinstance(stringlist, str):
            return '"' + os.path.join(additional_path, stringlist) + '"'
        elif isinstance(stringlist, list):
            res = '"'
            for string in stringlist:
                res += additional_path + string + " "
            res += '"'
            return res
        else:
            raise TypeError("Type must be a list a string or a list of strings.")


def create_erosita_badpix_to_detmap(badpix_filename="tm1_badpix_140602v01.fits",
                                    detmap_filename="tm1_detmap_100602v02.fits",
                                    output_filename="new_tm1_detmap_140602v01.fits"):
    """
    Creates new detmaps for Erosita in which bad pixels are added to the detector map.
    To be run for ALL modules in the event file before getting exposure maps with bad pixel
    correction.


    Parameters:
    - badpix_filename (str): The filename of the Erosita bad pixel file. Default is
    "tm1_badpix_140602v01.fits".
    - detmap_filename (str): The filename of the detector map file. Default is
    "tm1_detmap_100602v02.fits".
    - output_filename (str): The filename of the output detector map file. Default is
    "new_tm1_detmap_140602v01.fits".

    Returns:
    - None
    """
    badpix_file = fits.open(badpix_filename)
    badpix = np.vstack(badpix_file[1].data)
    badpix_subselection = badpix[:, :3].astype(np.int32) - 1
    hdulist = fits.open(detmap_filename)
    detmap = hdulist[0].data
    x_fix = badpix_subselection[:, 0]
    y_fix_start = badpix_subselection[:, 1]
    y_fix_end = y_fix_start + badpix_subselection[:, 2] + 1
    for i in range(badpix_subselection[0].shape[0] - 1):
        mask = (slice(y_fix_start[i], y_fix_end[i]), x_fix[i])
        detmap[mask] = 0
    import matplotlib.pyplot as plt
    plt.imshow(detmap)
    plt.show()
    hdulist.writeto(output_filename)
    hdulist.close()
