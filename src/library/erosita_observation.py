# FIXME: add copyright
import os
import subprocess
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import colors

from src.library.utils import check_type


class ErositaObservation:
    """
    Base class to retrieve and process eROSITA data.


    # Input datasets: eSASS event files or images with a full set of eSASS standard file extensions. FIXME

    """

    def __init__(self, input_filename, output_filename, working_directory):
        # TODO: Add parameter checks
        self.working_directory = os.path.abspath(working_directory)
        self.input = input_filename
        self.output = output_filename
        self.image = " erosita/esass:latest "

        # TODO: Add all fits file fields

        self._mounted_dir = "/home/idies/mnt_dir/"
        self._base_command = "docker run --platform linux/amd64 --volume " + self.working_directory + ":" + \
                             self._mounted_dir + \
                             self.image + "/bin/bash -c 'source ~/.bashrc && "

    def get_data(self, **kwargs):
        """
        Allows to extract and manipulate data from eROSITA event files through the eSASS 'evtool' command.

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
        print("The processed dataset has been saved as {}.".format(os.path.join(self.working_directory, self.output)))
        return fits.open(os.path.join(self.working_directory, self.output))

    def get_exposure_maps(self, template_image, emin, emax, **kwargs):
        """
        Computes exposure maps for eROSITA event files through the eSASS 'expmap' command.

        Parameters
        ----------
        template_image: str
        Path to the output exposure maps will be binned as specified in the WCS keywords of the template image.
        emin:
        emax:
        # TODO
        If withinputmaps=YES: input exposure maps
        """
        # TODO: parameter checks
        input_files = self._parse_stringlists(self.input, additional_path=self._mounted_dir)
        template_image = os.path.join(self._mounted_dir, template_image)
        flags = self._get_exmap_flags(self._mounted_dir, template_image, emin, emax, **kwargs)
        command = self._base_command + 'expmap ' + input_files + flags + "'"

        print(command)
        # exit()
        self._run_task(command)

    def get_psf(self, images, psfmaps, expimages, **kwargs):
        """
        Template image: the output exposure maps will be binned as specified in the WCS keywords of the template image.
        If withinputmaps=YES: input exposure maps
        """
        # TODO: parameter checks
        images = self._parse_stringlists(images, self._mounted_dir)
        psfmaps = os.path.join(self._mounted_dir, psfmaps)
        expimages = os.path.join(self._mounted_dir, expimages)
        flags = self._get_apetool_flags(images=images, psfmaps=psfmaps, expimages=expimages, **kwargs)
        command = self._base_command + 'apetool' + flags + "'"

        print(command)
        # exit()
        self._run_task(command)

    def _run_task(self, command):
        proc = subprocess.Popen([command], shell=True)
        proc.wait()
        (stdout, stderr) = proc.communicate()
        if proc.returncode != 0:
            raise FileNotFoundError("Docker Error")
        else:
            print("eSASS task COMPLETE.")

    def load_fits_data(self, filename):
        print("Loading ouput data stored in {}.".format(filename))
        return fits.open(os.path.join(self.working_directory, filename))

    def get_center_coordinates(self, input_filename):
        conv = 3600 # to arcseconds
        try:
            input_header = self.load_fits_data(input_filename)[1].header # FIXME: think about nicer implementation
            return conv*input_header['RA_PNT'], conv*input_header['DEC_PNT']
        except ValueError:
            print("Input filename does not contain center information.")
            return None

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
    def _get_evtool_flags(clobber=True, events=True, image=False, size=None,
                          rebin=None, center_position=None, region=None,
                          gti=None, flag=None, flag_invert=None, pattern=None,
                          telid=None, emin=None, emax=None, rawxy=None,
                          rawxy_telid=None, rawxy_invert=False, memset=None,
                          overlap=None, skyfield=None):

        input_params = {'clobber': bool, 'events': bool, 'image': bool, 'size': int,
                        'rebin': int, 'center_position': tuple, 'region': str,
                        'gti': str, 'flag': str, 'flag_invert': bool, 'pattern': int,
                        'telid': str, 'emin': float, 'emax': float, 'rawxy': str,
                        'rawxy_telid': int, 'rawxy_invert': bool, 'memset': int,
                        'overlap': float, 'skyfield': str}

        # Implements type checking
        for key, val in input_params.items():
            check_type(eval(key), val, name=key)

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
        flags += " center_position={}".format(center_position) if center_position is not None else ""
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
                         withsinglemaps=False, withmergedmaps=True, singlemaps=None,
                         mergedmaps=None, gtitype='GTI', withvignetting=True,
                         withdetmaps=False, withweights=True, withfilebadpix=True,
                         withcalbadpix=True, withinputmaps=False):

        input_params = {'mounted_dir': str, 'templateimage': str, 'emin': float, 'emax': float,
                        'withsinglemaps': bool, 'withmergedmaps': bool, 'singlemaps': list,
                        'mergedmaps': list, 'gtitype': str, 'withvignetting': bool,
                        'withdetmaps': bool, 'withweights': bool, 'withfilebadpix': bool,
                        'withcalbadpix': bool, 'withinputmaps': bool}

        # Implements type checking
        for key, val in input_params.items():
            check_type(eval(key), val, name=key)

        flags = " "
        flags += templateimage if templateimage is not None else print(
            "template image cannot be None.")  # FIXME Add exit somehow
        flags += " emin={}".format(emin) if emin is not None else print("emin cannot be None.")
        flags += " emax={}".format(emax) if emax is not None else print("emax cannot be None.")
        flags += " withsinglemaps=yes" if withsinglemaps else ""
        flags += "" if withmergedmaps else " withmergedmaps=no"
        flags += " singlemaps={}".format(os.path.join(mounted_dir, singlemaps)) if singlemaps is not None else ""
        flags += " mergedmaps={}".format(os.path.join(mounted_dir, mergedmaps)) if mergedmaps is not None else ""
        flags += " gtitype={}".format(gtitype) if gtitype != "GTI" else ""
        flags += "" if withvignetting else " withvignetting=no"
        flags += " withdetmaps=yes" if withdetmaps else ""
        flags += "" if withweights else " withweights=no"
        flags += "" if withfilebadpix else " withfilebadpix=no"
        flags += "" if withcalbadpix else " withcalbadpix=no"
        flags += " withinputmaps=yes" if withinputmaps else ""

        return flags

    @staticmethod
    def _get_apetool_flags(mllist: str = None, apelist: str = None, apelistout: str = None, images: [str] = None,
                           psfmaps: [str] = None, expimages: [str] = None, detmasks: [str] = None,
                           bkgimages: [str] = None, srcimages: [str] = None, apesenseimages: [str] = None,
                           emin: [float] = None, emax: [float] = None, eindex: [float] = None, eefextract: float = 0.7,
                           pthresh: float = 4e-6, cutrad: float = 15., psfmapsampling: float = 11.,
                           apexflag: bool = False, stackflag: bool = False, psfmapflag: bool = False,
                           shapepsf: bool = True, apesenseflag: bool = False):

        input_params = {'mllist': str, 'apelist': str, 'apelistout': str, 'images': list,
                        'psfmaps': list, 'expimages': list, 'detmasks': list,
                        'bkgimages': list, 'srcimages': list, 'apesenseimages': list,
                        'emin': list, 'emax': list, 'eindex': list, 'eefextract': float,
                        'pthresh': float, 'cutrad': float, 'psfmapsampling': float,
                        'apexflag': bool, 'stackflag': bool, 'psfmapflag': bool,
                        'shapepsf': bool, 'apesenseflag': bool}

        # Implements type checking
        for key, val in input_params.items():
            check_type(eval(key), val, name=key)

        flags = " "
        flags += " mllist={}".format(mllist) if mllist is not None else ""
        flags += " apelist={}".format(apelist) if apelist is not None else ""
        flags += " apelistout={}".format(apelistout) if apelistout is not None else ""
        flags += " images={}".format(images)
        flags += " psfmaps={}".format(psfmaps)
        flags += " expimages={}".format(expimages)
        flags += " detmasks={}".format(detmasks) if detmasks is not None else ""
        flags += " bkgimages={}".format(bkgimages) if bkgimages is not None else ""
        flags += " srcimages={}".format(srcimages) if srcimages is not None else ""
        flags += " apesenseimages={}".format(apesenseimages) if apesenseimages is not None else ""
        flags += " emin={}".format(emin) if emin is not None else ""
        flags += " emax={}".format(emax) if emax is not None else ""
        flags += " eindex={}".format(eindex) if eindex is not None else ""
        flags += " eefextract={}".format(eefextract) if eefextract != 0.7 else ""
        flags += " pthresh={}".format(pthresh) if pthresh != 4e-6 else ""
        flags += " cutrad={}".format(cutrad) if cutrad != 15. else ""
        flags += " psfmapsampling={}".format(psfmapsampling) if psfmapsampling != 11. else ""
        flags += " apexflag=yes" if apexflag else ""
        flags += " stackflag=yes" if stackflag else ""
        flags += " psfmapflag=yes" if psfmapflag else ""
        flags += "" if shapepsf else " shapepsf=no"
        flags += " apesenseflag=yes" if apesenseflag else ""

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


# if __name__ == "__main__":
#     cfg = xu.get_cfg("demos/eROSITA_config.yaml")
#     # File Location
#     file_info = cfg['files']
#     obs_path = file_info['obs_path']
#     input_filenames = file_info['input']
#     output_filename = file_info['output']
#     exposure_filename = file_info['exposure']
#     observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)
#
#     # Grid Info
#
#     grid_info = cfg['grid']
#     e_min = grid_info['energy_bin']['e_min']
#     e_max = grid_info['energy_bin']['e_max']
#     npix = grid_info['npix']
#
#     # Telescope Info
#
#     tel_info = cfg['telescope']
#     tm_id = tel_info['tm_id']
#
#
#     log = 'Output file {} already exists and is not regenerated. If the observations parameters shall be changed ' \
#           'please delete or rename the current output file.'
#
#     if not os.path.exists(os.path.join(obs_path, output_filename)):
#         observation = observation_instance.get_data(emin=e_min, emax=e_max, image=True, rebin=tel_info['rebin'],
#                                                     size=npix, pattern=tel_info['pattern'],
#                                                     telid=tm_id)
#     else:
#         print(log.format(os.path.join(obs_path, output_filename)))
#
#     observation_instance = ErositaObservation(output_filename, output_filename, obs_path)
#
#     # Exposure
#     if not os.path.exists(os.path.join(obs_path, exposure_filename)):
#         observation_instance.get_exposure_maps(output_filename, e_min, e_max, mergedmaps=exposure_filename)
#
#     else:
#         print(log.format(os.path.join(obs_path, output_filename)))
#     # Plotting
#     plot_info = cfg['plotting']
#     if plot_info['enabled']:
#         observation_instance.plot_fits_data(output_filename,
#                                             os.path.splitext(output_filename)[0],
#                                             slice=tuple(plot_info['slice']),
#                                             dpi=plot_info['dpi'])
#         observation_instance.plot_fits_data(exposure_filename,
#                                             f'{os.path.splitext(exposure_filename)[0]}.png',
#                                             slice=tuple(plot_info['slice']),
#                                             dpi=plot_info['dpi'])
#
#     data = observation_instance.load_fits_data(output_filename)[0].data
