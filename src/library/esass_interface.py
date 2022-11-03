# FIXME: Add copyright statement
# import docker
#

#     working_directory = {'/Users/matteani/Desktop': {'bind': '/mnt_dir', 'mode': 'rw'}}
#
#     client = docker.from_env()
#     container = client.containers.run("erosita/esass:latest", ["/bin/bash", "-c",
#                                                                "source ~/.bashrc && ls && evtool "
#                                                                "/home/idies/mnt_dir/LMC_SN1987A/fm00_700203_020_EventList_c001.fits /home/idies/mnt_dir/results.fits"],
#                                       detach=False, volumes=working_directory, stderr=True, stdout=True)
#     print(container.logs())

# ["source ~/.bashrc", "evtool /home/idies/mnt_dir/LMC_SN1987A/fm00_700203_020_EventList_c001.fits
# /home/idies/mnt_dir/results.fits"]

# OR

# ---------------------
# from python_on_whales import docker
#
# if __name__ == "__main__":
#     working_directory = "/Users/matteani/Desktop"
#     mounted_directory = "/home/idies/mnt_dir/"
#     command = "evtool " + mounted_directory + "LMC_SN1987A/fm00_700204_020_EventList_c001.fits " +
#     mounted_directory + "blabla.fits"
#     # command = "pwd"
#     output = docker.run("erosita/esass:latest", ["/bin/bash", "-c", "source ~/.bashrc &&", command],
#     interactive=True, volumes=[(working_directory, mounted_directory)], detach=False)
#     print(output)


import os
import subprocess


# FIXME PROVISIONARY (remove old docker interfaces and check)


class EsassInterface:
    """
    Base class to provide an interface with the Esass tools.

    """

    def __init__(self, working_directory, dataset, output_filename):
        self.working_directory = os.path.abspath(
            working_directory)  # FIXME: make sure that working dir coincides with where the output is gathered from
        # the erosita observation
        self.dataset = dataset
        self.output = output_filename
        self.image = " erosita/esass:latest "
        self._mounted_dir = "/home/idies/mnt_dir/"
        self._base_command = None

    def run_task(self, task, **kwargs):
        input_file = self._mounted_dir + self.dataset
        output_file = self._mounted_dir + self.output

        self._base_command = "docker run --platform linux/amd64 --volume " + self.working_directory + ":" + \
                             self._mounted_dir + \
                             self.image + "/bin/bash -c 'source ~/.bashrc && "
        match task:  # FIXME
            case "evtool":
                command = self._run_evtool(event_files=input_file, outfile=output_file, **kwargs)
            case "expmap":
                flags = self._run_expmap()
            case other:
                print("Invalid or not yet implemented esass task.")
                exit()

        proc = subprocess.Popen([command], shell=True)
        proc.wait()
        (stdout, stderr) = proc.communicate()
        if proc.returncode != 0:
            print(stderr)
        else:
            print(
                "The processed dataset has been saved as {}.".format(os.path.join(self.working_directory, self.output)))

    def _run_evtool(self, event_files: list[str], outfile: str, clobber: bool = True, events: bool = True,
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

        command = self._base_command + 'evtool ' + event_files + " " + outfile + flags + "'"
        print(command)
        exit()
        return command

    def _run_expmap(self, inputdatasets: list[str], templateimage: str, emin: float, emax: float,
                     withsinglemaps: bool = False, withmergedmaps: bool = True, singlemaps: list[str] = None,
                     mergedmaps: list[str] = None, gtitype: str = 'GTI', withvignetting: bool = True,
                     withdetmaps: bool = False, withweights: bool = True, withfilebadpix: bool = True,
                     withcalbadpix: bool = True, withinputmaps: bool = False):
        pass


if __name__ == "__main__":
    working_directory = "../../../data/"
    data = "LMC_SN1987A/fm00_700203_020_EventList_c001.fits"
    int = EsassInterface(working_directory, data, "output.fits")
    int.run_task(task='evtool', emin=0.2, emax=0.5, image=True, rebin=80, size=3240, pattern=15)
