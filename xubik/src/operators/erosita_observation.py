# FIXME: add copyright
import os.path
import subprocess

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors


class ErositaObservation:

    """
    Base class to retrieve and process eROSITA data.


    #Input datasets: eSASS event files or images with a full set of eSASS standard file extensions. FIXME

    """

    def __init__(self, input_filename: str, output_filename: str):
        # TODO: Add parameter checks
        self.working_directory = os.path.abspath("../../../data/") #FIXME: maybe let this be decided by the user
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
        # event_files: list[str], outfile: str, FIXME
        """
        print("Collecting data from {}.". format(self.input))
        input_file = self._mounted_dir + self.input
        output_file = self._mounted_dir + self.output

        flags = self._get_evtool_flags(**kwargs)
        command = self._base_command + 'evtool ' + input_file + " " + output_file + flags + "'"
        self._run_task(command)
        print("The processed dataset has been saved as {}.".format(os.path.join(self.working_directory, self.output)))
        return fits.open(os.path.join(self.working_directory, self.output))

    def load_data(self, filename):
        print("Loading ouput data stored in {}.".format(filename))
        print(os.path.join(self.working_directory, filename))
        return fits.open(os.path.join(self.working_directory, filename))

    def get_exposure_maps(self, template_image, emin, emax, **kwargs):
        """
        Template image: the output exposure maps will be binned as specified in the WCS keywords of the template image.
        If withinputmaps=YES: input exposure maps
        """
        input_file = self._mounted_dir + self.input
        template_image = self._mounted_dir + template_image
        flags = self._get_exmap_flags(template_image, emin, emax, **kwargs)
        command = self._base_command + 'expmap ' + input_file + flags + "'"
        # command = command = self._base_command + 'cd ' + self._mounted_dir + " && ls" "'"
        # print(command)
        # exit()
        self._run_task(command)
        exit()

    def _run_task(self, command):
        proc = subprocess.Popen([command], shell=True)
        proc.wait()
        (stdout, stderr) = proc.communicate()
        if proc.returncode != 0:
            print(stderr)
        else:
            print("eSASS task COMPLETE.")

    @staticmethod
    def _get_evtool_flags(clobber: bool = True, events: bool = True,
                          image: bool = False, size: list[str] = None, rebin: list[str] = None,
                          center_position: list[str] = None, region: str = None, gti: list[str] = None, flag: str = None,
                          flag_invert: bool = None, pattern: int = None, telid: list[str] = None, emin: list[str] = None,
                          emax: list[str] = None, rawxy: str = None, rawxy_telid: int = None, rawxy_invert: bool = False,
                          memset: int = None, overlap: float = None, skyfield: str = None):

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
    def _get_exmap_flags(templateimage: str, emin: float, emax: float,
                     withsinglemaps: bool = False, withmergedmaps: bool = True, singlemaps: list[str] = None,
                     mergedmaps: list[str] = None, gtitype: str = 'GTI', withvignetting: bool = True,
                     withdetmaps: bool = False, withweights: bool = True, withfilebadpix: bool = True,
                     withcalbadpix: bool = True, withinputmaps: bool = False):

        flags = " "
        flags += templateimage if templateimage is not None else print("template image cannot be None.") # FIXME Add exit somehow
        flags += " emin={}".format(emin) if emin is not None else print("emin cannot be None.")
        flags += " emax={}".format(emax) if emax is not None else print("emax cannot be None.")
        flags += " withsinglemaps=yes" if withsinglemaps else ""
        flags += "" if withmergedmaps else " withmergedmaps=no"
        flags += " singlemaps={}".format(singlemaps) if singlemaps is not None else ""
        flags += " mergedmaps={}".format(mergedmaps) if mergedmaps is not None else ""
        flags += " gtitype={}".format(gtitype) if gtitype != "GTI" else ""
        flags += "" if withvignetting else " withvignetting=no"
        flags += " withdetmaps=yes" if withdetmaps else ""
        flags += "" if withweights else " withweights=no"
        flags += "" if withfilebadpix else " withfilebadpix=no"
        flags += "" if withcalbadpix else " withcalbadpix=no"
        flags += " withinputmaps=yes" if withinputmaps else ""

        return flags


if __name__ == "__main__":
    obs_path = "../../../data/"
    filename = "output.fits"
    input_filename = "LMC_SN1987A/fm00_700203_020_EventList_c001.fits"

    output_filename = obs_path + os.path.splitext(filename)[0] + ".png"

    observation_instance = ErositaObservation(input_filename, "out.fits")
    # observation_instance.get_exposure_maps("out.fits", 0.2, 0.5)
    observation = observation_instance.get_data(emin=0.2, emax=0.5, image=True, rebin=80, size=3240, pattern=15)
    observation = observation_instance.load_data("out.fits")
    data = observation[0].data

    plt.imshow(data, origin='lower', norm=colors.SymLogNorm(linthresh=8 * 10e-3))
    plt.savefig(output_filename)
    print("Data saved as {}.".format(output_filename))








