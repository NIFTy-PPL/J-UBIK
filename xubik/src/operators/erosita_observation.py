# FIXME: add copyright
import os.path
import subprocess

import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import colors


class ErositaObservation:
    """
    Base class to retrieve and process eROSITA data.


    # Input datasets: eSASS event files or images with a full set of eSASS standard file extensions. FIXME

    """

    def __init__(self, input_filename: list[str], output_filename: str):
        # TODO: Add parameter checks
        self.working_directory = os.path.abspath("../../../data/")  # FIXME: maybe let this be decided by the user
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
        # print(command)
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
            print(stderr)
        else:
            print("eSASS task COMPLETE.")

    def load_data(self, filename):
        print("Loading ouput data stored in {}.".format(filename))
        print(os.path.join(self.working_directory, filename))
        return fits.open(os.path.join(self.working_directory, filename))

    @staticmethod
    def _get_evtool_flags(clobber: bool = True, events: bool = True, image: bool = False, size: list[str] = None,
                          rebin: list[str] = None, center_position: list[str] = None, region: str = None,
                          gti: list[str] = None, flag: str = None, flag_invert: bool = None, pattern: int = None,
                          telid: list[str] = None, emin: list[str] = None, emax: list[str] = None, rawxy: str = None,
                          rawxy_telid: int = None, rawxy_invert: bool = False, memset: int = None,
                          overlap: float = None, skyfield: str = None):
        """
        Returns appropriate evtool command flags.

        Parameters
        ----------

        clobber: bool
        events: list[str]
        image: bool
        size: list[str]
        rebin: list[str]
        center_position: list[str]
        region: str
        gti: list[str]
        flag: str
        flag_invert: bool
        pattern: int
        telid: list[str]
        emin: list[str]
        emax: list[str]
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
                         withsinglemaps: bool = False, withmergedmaps: bool = True, singlemaps: list[str] = None,
                         mergedmaps: list[str] = None, gtitype: str = 'GTI', withvignetting: bool = True,
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
    def _parse_stringlists(stringlist: list[str], additional_path: str = ""):
        res = '"'
        for string in stringlist:
            res += additional_path + string + " "
        res += '"'
        return res


if __name__ == "__main__":
    obs_path = "../../../data/" # Folder that gets mounted to the docker
    filename = "combined_out_08_1.fits"
    input_filename = ['LMC_SN1987A/fm00_700203_020_EventList_c001.fits', 'LMC_SN1987A/fm00_700204_020_EventList_c001.fits', 'LMC_SN1987A/fm00_700204_020_EventList_c001.fits']
    # input_filename = "LMC_SN1987A/fm00_700203_020_EventList_c001.fits"
    # input_filename = "Vela_SNR/pm00_700039_020_EventList_c001.fits"
    # output_filename = obs_path + os.path.splitext(input_filename)[0] + ".png"

    observation_instance = ErositaObservation(input_filename, filename)
    # observation = observation_instance.get_data(emin=0.2, emax=0.5, image=True, rebin=80, size=3240, pattern=15)
    # observation = observation_instance.get_data(emin=0.7, emax=1.0, image=True, rebin=80, size=3240, pattern=15)
    observation_instance.get_exposure_maps(filename, 0.7, 1.0, mergedmaps="expmap_combined.fits")


    # observation_instance.get_exposure_maps("out_2.fits", 0.2, 0.5, mergedmaps="expmap.fits")
    # observation_instance.get_psf("out_2.fits", "psfmaps_2.fits", "expmap.fits", psfmapflag=True)
    # observation = observation_instance.load_data("output_vela.fits")
    observation = observation_instance.load_data("combined_out_08_1.fits")

    # print(observation)
    # exit()
    data = observation[0].data
    # expmap = observation_instance.load_data("expmap.fits")[0].data
    # psfmap = observation_instance.load_data("psfmaps_2.fits")
    # psfmap = observation_instance.load_data("tm3_2dpsf_100215v02.fits")
    # psfmap = psfmap[0].dataR
    # print(repr(psfmap))
    # exit()


    plt.imshow(data, origin='lower', norm=colors.SymLogNorm(linthresh=1e-1))
    # plt.imshow(expmap, origin='lower', norm=colors.SymLogNorm(linthresh=8 * 10e-3))
    # plt.imshow(psfmap, origin='lower', norm=colors.SymLogNorm(linthresh=8 * 10e-8, vmax=1e-4))
    # plt.plot
    plt.colorbar()
    plt.show()

    # plt.savefig(output_filename)
    # plt.show()
    # exit()

    # def show_images(images):
    #     n: int = len(images)
    #     f = plt.figure()
    #     for i in range(n):
    #         # Debug, plot figure
    #         f.add_subplot(3, 4, i + 1)
    #         plt.imshow(images[i])
    #
    #     # plt.show(block=True)
    #     name = os.path.splitext(output_filename)[0] + "_psf.png" #FIXME
    #     plt.savefig(os.path.splitext(output_filename)[0] + "_psf.png")
    #
    # show_images(psfmap)

    # print("Data saved as {}.".format(output_filename))