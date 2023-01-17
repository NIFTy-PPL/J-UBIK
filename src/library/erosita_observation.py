# FIXME: add copyright
import os
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import colors


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

    def plot_fits_data(self, filename, image_name, slice=None, lognorm=True, linthresh=10e-1,
                       show=False, dpi=None, **kwargs):
        im = self.load_fits_data(filename)[0].data
        if slice is not None:
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
    def _get_evtool_flags(clobber: bool = True, events: bool = True, image: bool = False, size=None,
                          rebin=None, center_position=None, region: str = None,
                          gti=None, flag: str = None, flag_invert: bool = None, pattern: int = None,
                          telid=None, emin=None, emax=None, rawxy: str = None,
                          rawxy_telid: int = None, rawxy_invert: bool = False, memset: int = None,
                          overlap: float = None, skyfield: str = None):
        """
        Returns appropriate evtool command flags.

        Parameters
        ----------

        clobber: bool
        events
        image: bool
        size
        rebin
        center_position
        region: str
        gti
        flag: str
        flag_invert: bool
        pattern: int
        telid
        emin
        emax
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
    def _get_exmap_flags(mounted_dir: str, templateimage: str, emin: float, emax: float,
                         withsinglemaps: bool = False, withmergedmaps: bool = True, singlemaps=None,
                         mergedmaps=None, gtitype: str = 'GTI', withvignetting: bool = True,
                         withdetmaps: bool = False, withweights: bool = True, withfilebadpix: bool = True,
                         withcalbadpix: bool = True, withinputmaps: bool = False):

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("obs_path", type=str, nargs='?', default="../data/LMC_SN1987A/")
    parser.add_argument("plotting", type=bool, nargs='?', default=False)
    args = parser.parse_args()
    obs_path = args.obs_path  # Folder that gets mounted to the docker
    input_filenames = ['fm00_700203_020_EventList_c001.fits']
    output_filename = "combined_out_08_1_no_imm.fits"
    observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)

    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation = observation_instance.get_data(emin=1.0, emax=2.3, image=True, rebin=80, size=3240, pattern=15,
                                                    telid=1)
    else:
        print(
            'Output file already exists and is not regenerated. If the observations parameters shall be changed please'
            'delete or rename the current output file.')

    observation_instance = ErositaObservation(output_filename, output_filename, obs_path)
    # Exposure
    observation_instance.get_exposure_maps(output_filename, 0.7, 1.0, mergedmaps="expmap_combined.fits")
    if args.plotting:
        observation_instance.plot_fits_data(output_filename, "combined_out_08_1_no_imm", slice=(1100, 2000, 800, 2000),
                                            dpi=800)
        observation_instance.plot_fits_data("expmap_combined.fits", "expmap_combined.png",
                                            slice=(1100, 2000, 800, 2000), dpi=800)

    data = observation_instance.load_fits_data(output_filename)
