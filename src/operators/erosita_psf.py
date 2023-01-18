import numpy as np
import astropy.io.fits as ast
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

dir_path= "tm1/bcf/"
fname = ["tm1_2dpsf_190219v05.fits", "tm1_2dpsf_190220v03.fits"] # PSF-correction / Data modeling

    # "tm1_2dpsf_100215v02.fits", "tm1_2dpsf_110705v01.fits"containing ,
    # "tm1_2dpsf_190219v03.fits", "tm1_2dpsf_190219v04.fits",

class eROSITA_PSF():
    """
    fname: Filename / Path of the psf.fits file.
    """
    def __init__(self, fname):
        self._fname = fname

    def _load_fits(self):
        return ast.open(self._fname)

    def _load_names(self):
        with ast.open(self._fname) as f:
            name_list = [f[i].name for i in range(len(f))]
        return name_list

    def _ind_for_energy(self, energy):
        """Energy: String, e.g. 1000eV"""
        cut_list = []
        with ast.open(self._fname) as f:
            for i in range(len(f)):
                if energy in f[i].name:
                    cut_list.append(i)
        return cut_list

    def _load_data(self, energy):
        """Energy: String, e.g. 1000eV"""
        ind = self._ind_for_energy(energy)
        with ast.open(self._fname) as f:
            data_list = [f[i].data for i in ind]
        return np.array(data_list)

    def _load_data_full(self):
        """PSFs"""
        with ast.open(self._fname) as f:
            data_list = [f[i].data for i in range(len(f))]
        return np.array(data_list)

    def _load_p_center(self, energy):
        "Origin of PSF in Pixel Values"
        ind = self._ind_for_energy(energy)
        with ast.open(self._fname) as f:
            p_center = [(f[i].header["CRPIX1"], f[i].header["CRPIX2"]) for i in ind]
        return np.array(p_center)

    def _load_p_center_full(self):
        "Origin of PSF in Pixel Values"
        with ast.open(self._fname) as f:
            p_center = [(f[i].header["CRPIX1"], f[i].header["CRPIX2"]) for i in range(len(f))]
        return np.array(p_center)

    def _load_pix_size(self):
        """Pixel Size in arcsecs"""
        with ast.open(self._fname) as f:
            p_size = [(f[0].header["CDELT1"], f[0].header["CDELT2"])]
        return np.array(p_size)

    def _load_pix_size_full(self):
        with ast.open(self._fname) as f:
            p_size = [(f[i].header["CDELT1"], f[i].header["CDELT2"]) for i in range(len(f))]
        return np.array(p_size)

    def _load_theta(self, energy):
        ind = self._ind_for_energy(energy)
        with ast.open(self._fname) as f:
            theta_list = [int(f[i].name.split("a")[0].split("V")[1]) for i in ind]
        return np.array(theta_list)*60

    def _load_theta_full(self):
        with ast.open(self._fname) as f:
            theta_list = [int(f[i].name.split("a")[0].split("V")[1]) for i in range(len(f))]
        return np.array(theta_list)*60

    def _load_energy(self):
        with ast.open(self._fname) as f:
            energy_list = [f[i].name.split("e")[0] for i in range(len(f))]
        return energy_list

    def _load_theta_boundaries(self):
        with ast.open(self._fname) as f:
            theta_list = [f[i].header["CBD20001"] for i in range(len(f))]
        return theta_list

    def _load_energy_boundaries(self):
        with ast.open(self._fname) as f:
            theta_list = [f[i].header["CBD10001"] for i in range(len(f))]
        return theta_list

    def info(self, energy):
        full_dct = {
            "psf": self._load_data(energy),
            "theta": self._load_theta(energy),
            "center": self._load_p_center(energy),
            "dpix": self._load_pix_size()}
        return full_dct

    def plot_psfs(self):
        """plots the psfs in the fits file"""
        name = self._load_names()
        psf = self._load_data_full()
        theta = self._load_theta_full()
        center = self._load_p_center_full()
        obj = zip(name, psf, theta, center)
        for _, j in enumerate(obj):
            fig, axs = plt.subplots()
            axs.set_title(f"{j[0]} point_source at {j[3]}")
            im = axs.imshow(j[1], norm=LogNorm(), origin="lower")
            axs.scatter(j[3][0], j[3][1], marker="x")
            axs.set_xlabel('[arcsec]')
            axs.set_ylabel('[arcsec]')
            fig.colorbar(mappable=im)
            plt.savefig(f'psf_{j[0]}.png')
            plt.clf
            plt.close()

a = eROSITA_PSF(dir_path+fname[4])
a.plot_psfs()
