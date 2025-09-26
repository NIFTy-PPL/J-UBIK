# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

import os
import subprocess
from os.path import join
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colors

from ...utils import _check_type


class ErositaObservation:
    """
    Base class for retrieving and processing eROSITA data using eSASS commands.

    This class facilitates the retrieval and manipulation of eROSITA event
    files and images.
    It assumes the presence of standard eSASS file extensions and utilizes
    Docker to run eSASS commands in a containerized environment.
    For more information on the eSASS software and installation, see
    <https://erosita.mpe.mpg.de/dr1/eSASS4DR1/>

    Attributes
    ----------
    working_directory : str
        Absolute path to the directory where output files will be stored.
    input : str
        Filename of the input eROSITA event file or image.
    output : str
        Filename for the output file.
    image : str
        Docker image to use for running eSASS commands.
        Defaults to the latest EDR or DR1 image.
    _mounted_dir : str
        Directory path inside the Docker container where the working directory
        is mounted.
    _base_command : str
        Base command for running eSASS commands in the Docker container.

    """

    def __init__(self, input_filename, output_filename, working_directory,
                 esass_image=None):
        """
        Initializes the ErositaObservation class.

        Parameters
        ----------
        input_filename : str
            Filename of the input eROSITA event file or image.
        output_filename : str
            Filename for the output file.
        working_directory : str
            Directory where output files will be stored.
        esass_image : str, optional
        Docker image tag to use for running eSASS commands.
        Options are 'EDR' or 'DR1'. Defaults to 'EDR'.
        If not provided, defaults to the latest EDR image.

        Raises
        ------
        ValueError
            If `esass_image` is provided but does not match 'EDR' or 'DR1'.

        Notes
        -----
        - Ensure Docker is installed and properly configured on the system.
        - This class assumes that eSASS commands are run in a Docker container
        with the specified image and that the eSASS environment is correctly
        set up within the container.
        """
        # TODO: Add parameter checks
        self.working_directory = os.path.abspath(working_directory)
        self.input = input_filename
        self.output = output_filename
        if esass_image is None or esass_image == "EDR":
            self.image = " erosita/esass:latest "
        elif esass_image == "DR1":
            self.image = " erosita/esass-x64:latest "
        else:
            raise ValueError(
                f"esass_image must be either EDR or DR1, got {esass_image}.")

        # TODO: Add all fits file fields

        self._mounted_dir = "/home/idies/mnt_dir/"
        self._base_command = "docker run --platform linux/amd64 --volume " + \
                             self.working_directory + ":" + \
                             self._mounted_dir + \
                             self.image + "/bin/bash -c 'source ~/.bashrc && "

    def get_data(self,
                 pointing_center: Union[tuple, list],
                 **kwargs):
        """
        Extracts and manipulates data from eROSITA event files using the eSASS
        `evtool` and `radec2xy` commands.

        This method constructs and executes a command to process eROSITA event
        files and save the resulting dataset. The command is executed with
        options specified through keyword arguments.

        Parameters
        ----------
        pointing_center: tuple or list
            The coordinates of the pointing center in the format
            (ra, dec) or [ra, dec].
        **kwargs : keyword arguments
            Additional arguments passed to the `_get_evtool_flags` method to
            customize the behavior of the 'evtool' command.

        Returns
        -------
        astropy.io.fits.HDUList
            An HDU list object containing the processed dataset, which is saved
            to the output file path specified.

        Notes
        -----
        - Ensure that the eSASS software is correctly installed and configured.
        - The output file is saved in the directory specified by
        `self.working_directory` and named according to `self.output`.
        - This method prints messages about the status of the data collection
        and saving process.
        """
        print(f"Collecting data from {self.input}.")
        input_files = self._parse_stringlists(self.input,
                                              additional_path=self._mounted_dir)
        output_file = self._mounted_dir + self.output
        ra, dec = pointing_center

        center_events_task = (f"{self._base_command}radec2xy"
                              f" {input_files} '{ra}' '{dec}' '")

        self._run_task(center_events_task)

        flags = self._get_evtool_flags(**kwargs)
        command = (self._base_command + 'evtool ' + input_files + " " +
                   output_file + flags + "'")

        self._run_task(command)
        print(f"The processed dataset has been saved "
              f"as {join(self.working_directory, self.output)}.")
        return fits.open(join(self.working_directory, self.output))

    def get_exposure_maps(self, template_image, emin, emax,
                          badpix_correction=True, **kwargs):
        """
        Computes exposure maps for eROSITA event files using the eSASS
        'expmap' command.

        This method constructs and executes the command to generate exposure
        maps based on the provided template image and energy range.
        If bad pixel correction is enabled, it updates the detector maps
        before running the exposure map computation.

        Parameters
        ----------
        template_image : str
            Path to the template image file.
            The exposure maps will be binned according
            to the WCS keywords of this template image.
        emin : float
            Minimum energy in keV for the exposure map computation.
        emax : float
            Maximum energy in keV for the exposure map computation.
        badpix_correction : bool, optional
            Whether to apply bad pixel correction to the detector maps.
            If  True, the method will update the detector maps using
            `create_erosita_badpix_to_detmap` before computing the
            exposure maps. Default is True.
        **kwargs : keyword arguments
            Additional arguments to pass to the `_get_exmap_flags` method.

        Returns
        -------
        None
            The method does not return a value.
            It executes a command to generate the exposure maps, which are
            saved to file. The output file is saved in the directory specified
            by `self.working_directory` and named according to
            `output_filename`.

        Notes
        -----
        - Ensure that the bad pixel correction files are available in the
        specified paths.
        - This method requires that the eSASS software is properly installed
        and configured.
        """
        # TODO: parameter checks
        input_files = self._parse_stringlists(self.input,
                                              additional_path=self._mounted_dir)
        template_image = join(self._mounted_dir, template_image)
        flags = self._get_exmap_flags(self._mounted_dir, template_image, emin,
                                      emax, **kwargs)
        command = self._base_command + 'expmap ' + input_files + flags + "'"

        caldb_loc_base = '/home/idies/caldb/data/erosita/tm{}/bcf/'
        detmap_file_base = 'tm{}_detmap_100602v02.fits'

        if badpix_correction:
            command = self._base_command
            for i in range(7):
                caldb_loc = caldb_loc_base.format(i + 1)
                detmap_file = detmap_file_base.format(i + 1)
                update_detmap_command = f' mv {caldb_loc}{detmap_file} {caldb_loc}' \
                                        f'tm{i + 1}_detmap_100602v02_old.fits && ' \
                                        f'cp {self._mounted_dir}new_detmaps/{detmap_file} {caldb_loc} &&'
                command += update_detmap_command
            command += ' expmap ' + input_files + flags + "'"

        self._run_task(command)

    def load_fits_data(self, filename, verbose=True):
        if verbose:
            print(f"Loading ouput data stored in {filename}.")
        return fits.open(join(self.working_directory, filename))

    def get_pointing_coordinates_stats(self, module, input_filename=None):
        if not isinstance(module, int):
            raise TypeError('Telescope module must be of type int (1-7).')
        module = 'CORRATT' + str(module)

        if input_filename is None:
            input_filename = self.input
        print(f'Loading pointing coordinates for TM{module[-1]} '
              f'module from {input_filename}.')

        try:
            data = self.load_fits_data(input_filename,
                                       verbose=False)[module].data

        except ValueError as err:
            raise ValueError(
                f"""
            Input filename does not contain pointing information.
            
            {err}
            """
            )

        # Convert pointing information to arcseconds
        ra = data['RA']
        dec = data['DEC']
        roll = data['ROLL']

        # Return pointing statistics
        stats = {'RA': (ra.mean(), ra.std()),
                 'DEC': (dec.mean(), dec.std()),
                 'ROLL': (roll.mean(), roll.std())}

        return stats

    def plot_fits_data(self, filename, image_name, slice=None, lognorm=True,
                       linthresh=10e-1, show=False, dpi=None, **kwargs):
        im = self.load_fits_data(filename)[0].data
        if slice is not None:
            slice = tuple(slice)
            im = im[slice[2]:slice[3], slice[0]:slice[1]]
        output = join(self.working_directory, image_name)
        norm = None
        if lognorm:
            norm = colors.SymLogNorm(linthresh=linthresh)
        plt.imshow(im, origin='lower', norm=norm, **kwargs)
        plt.colorbar()
        if show:
            plt.show()
        plt.savefig(output, dpi=dpi)
        plt.close()
        print(f"Plot from fits data saved as {output}.")

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

        input_params = {'clobber': bool,
                        'events': bool,
                        'image': bool,
                        'size': int,
                        'rebin': int,
                        'center_position': tuple,
                        'region': str,
                        'gti': str,
                        'flag': str,
                        'flag_invert': bool,
                        'pattern': int,
                        'telid': int,
                        'emin': float | str,
                        'emax': float | str,
                        'rawxy': str,
                        'rawxy_telid': int,
                        'rawxy_invert': bool,
                        'memset': int,
                        'overlap': float,
                        'skyfield': str}

        # Implements type checking
        for key, val in input_params.items():
            _check_type(eval(key), val, name=key)


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
        flags += " rawxy_telid={}".format(
            rawxy_telid) if rawxy_telid is not None else ""
        flags += " rawxy_invert=yes" if rawxy_invert else ""
        flags += " memset={}".format(memset) if memset is not None else ""
        flags += " overlap={}".format(overlap) if overlap is not None else ""
        flags += " skyfield={}".format(skyfield) if skyfield is not None else ""

        return flags

    @staticmethod
    def _get_exmap_flags(mounted_dir, templateimage, emin, emax,
                         withsinglemaps=False, withmergedmaps=False,
                         singlemaps=None, mergedmaps=None, gtitype='GTI',
                         withvignetting=True, withdetmaps=True,
                         withweights=True, withfilebadpix=True,
                         withcalbadpix=True, withinputmaps=False):

        input_params = {'mounted_dir': str,
                        'templateimage': str,
                        'emin': float | str,
                        'emax': float | str,
                        'withsinglemaps': bool,
                        'withmergedmaps': bool,
                        'singlemaps': list,
                        'mergedmaps': str,
                        'gtitype': str,
                        'withvignetting': bool,
                        'withdetmaps': bool,
                        'withweights': bool,
                        'withfilebadpix': bool,
                        'withcalbadpix': bool,
                        'withinputmaps': bool}

        # Implements type checking
        for key, val in input_params.items():
            _check_type(eval(key), val, name=key)

        if singlemaps is not None:
            singlemaps = list(map(lambda x: join(mounted_dir, x), singlemaps))
            singlemaps_string = '"' + " ".join(singlemaps) + '"'
        else:
            singlemaps_string = " "

        flags = " "
        flags += templateimage if templateimage is not None else print(
            "template image cannot be None.")  # FIXME Add exit somehow
        flags += " emin={}".format(emin) if emin is not None else print(
            "emin cannot be None.")
        flags += " emax={}".format(emax) if emax is not None else print(
            "emax cannot be None.")
        flags += " withsinglemaps=yes" if withsinglemaps else ""
        flags += "" if withmergedmaps else " withmergedmaps=no"
        flags += f" singlemaps={singlemaps_string}" if singlemaps is not None \
            else ""
        flags += " mergedmaps={}".format(
            join(mounted_dir, mergedmaps)) if mergedmaps is not None else ""
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
            return '"' + join(additional_path, stringlist) + '"'
        elif isinstance(stringlist, list):
            res = '"'
            for string in stringlist:
                res += additional_path + string + " "
            res += '"'
            return res
        else:
            raise TypeError(
                "Type must be a list a string or a list of strings.")


def create_erosita_badpix_to_detmap(
    badpix_filename="tm1_badpix_140602v01.fits",
    detmap_filename="tm1_detmap_100602v02.fits",
    output_filename="new_tm1_detmap_140602v01.fits"):
    """
    Creates a new detector map (detmap) for eROSITA by incorporating bad pixels
    into the existing detector map. This process should be performed for all
    modules in the event file before generating exposure maps with bad pixel
    correction.

    Parameters
    ----------
    badpix_filename : str, optional
        The filename of the eROSITA bad pixel file. Default is
        "tm1_badpix_140602v01.fits".
    detmap_filename : str, optional
        The filename of the detector map file. Default is
        "tm1_detmap_100602v02.fits".
    output_filename : str, optional
        The filename for the output detector map with the bad pixels included.
        Default is "new_tm1_detmap_140602v01.fits".

    Returns
    -------
    None
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
