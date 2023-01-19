import numpy as np
import astropy.io.fits as ast
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .psf_interpolation import get_psf_func, psf_convolve_operator, psf_lin_int_operator



# TODO If the energy is unclear to the user this throw and error
# + should print all available energy bins to the user.
class eROSITA_PSF():
    """
    fname: Filename / Path of the psf.fits file.
    """
    def __init__(self, fname):
        self._fname = fname
        # TODO: verify that this is the correct assignment!
        self._myheader = {'ra': "1", 'dec': "2"}

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
            p_center = [(f[i].header["CRPIX"+self._myheader['ra']], 
                         f[i].header["CRPIX"+self._myheader['dec']]) 
                         for i in ind]
        return np.array(p_center)

    def _load_p_center_full(self):
        "Origin of PSF in Pixel Values"
        with ast.open(self._fname) as f:
            p_center = [(f[i].header["CRPIX"+self._myheader['ra']], 
                         f[i].header["CRPIX"+self._myheader['dec']]) 
                         for i in range(len(f))]
        return np.array(p_center)

    def _load_pix_size(self):
        """Pixel Size in arcsecs"""
        with ast.open(self._fname) as f:
            p_size = [(f[0].header["CDELT"+self._myheader['ra']], 
                       f[0].header["CDELT"+self._myheader['dec']])]
        return np.array(p_size)

    def _load_pix_size_full(self):
        with ast.open(self._fname) as f:
            p_size = [(f[i].header["CDELT"+self._myheader['ra']], 
                       f[i].header["CDELT"+self._myheader['dec']]) 
                       for i in range(len(f))]
        return np.array(p_size)

    def _load_theta(self, energy):
        ind = self._ind_for_energy(energy)
        with ast.open(self._fname) as f:
            theta_list = [int(f[i].name.split("a")[0].split("V")[1]) 
                          for i in ind]
        return np.array(theta_list)*60

    def _load_theta_full(self):
        with ast.open(self._fname) as f:
            theta_list = [int(f[i].name.split("a")[0].split("V")[1]) 
                          for i in range(len(f))]
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

    def _cutnorm(self, psf, lower_cut = 1E-5, want_frac = False):
        if len(psf.shape) != 2:
            raise ValueError
        if want_frac:
            norm = np.sum(psf)
        if lower_cut is not None:
            psf[psf <= lower_cut] = 0.
        if want_frac:
            frac = np.sum(psf) / norm
        psf /= psf.sum()
        if want_frac:
            return psf, frac
        return psf

    def info(self, energy):
        full_dct = {
            "psf": self._load_data(energy),
            "theta": self._load_theta(energy),
            "center": self._load_p_center(energy),
            "dpix": self._load_pix_size()}
        return full_dct

    def plot_psfs(self, lower_cut = 1E-5):
        """plots the psfs in the fits file"""
        name = self._load_names()
        psf = self._load_data_full()
        theta = self._load_theta_full()
        center = self._load_p_center_full()
        obj = zip(name, psf, theta, center)
        for _, j in enumerate(obj):
            fig, axs = plt.subplots(figsize = (10,10))
            axs.set_title(f"{j[0]} point_source at {j[3]}")
            tm, frac = self._cutnorm(j[1], lower_cut=lower_cut, want_frac=True)
            axs.text(10, 450, f"Norm. fraction: {frac}")
            im = axs.imshow(tm.T, norm=LogNorm(), origin="lower")
            axs.scatter(j[3][0], j[3][1], marker=".", color='r')
            axs.set_xlabel('[arcsec]')
            axs.set_ylabel('[arcsec]')
            fig.colorbar(mappable=im)
            plt.savefig(f'psf_{j[0]}.png')
            plt.clf
            plt.close()

    def _get_obs_infos(self, energy, pointing_center, lower_cut = 1E-5):
        newpsfs = np.array([self._cutnorm(pp, lower_cut = lower_cut) for pp in 
                            self._load_data(energy)])
        obs_infos = {'psfs' : newpsfs, 
                     'rs' : self._load_theta(energy), 
                     'patch_center_ids' : self._load_p_center(energy),
                     'patch_deltas' : self._load_pix_size(), 
                     'pointing_center' : pointing_center}
        return obs_infos

    def make_psf_op(self, energy, pointing_center, domain, lower_radec, 
                    conv_method, conv_params):
        obs_infos = self._get_obs_infos(energy, pointing_center)

        if conv_method == 'MSC':
            print('Build MSC-PSF...')
            op = psf_convolve_operator(domain, lower_radec, obs_infos,
                                       conv_params)
            print('...done build MSC-PSF')
        elif conv_method == 'LIN':
            print('Build LIN-PSF...')
            op = psf_lin_int_operator(domain, conv_params['npatch'], 
                                      lower_radec, obs_infos,
                                      margfrac = conv_params['margfrac'],
                                      want_cut = conv_params['want_cut'])
            print('...done build LIN-PSF')
        else:
            raise ValueError(f'Unknown conv_method: {conv_method}')
        return op

    def _get_psf_func(self, energy, pointing_center, domain, lower_radec):
        obs_infos = self._get_obs_infos(energy, pointing_center)
        psf_func = get_psf_func(domain, lower_radec, obs_infos)
        return psf_func


    def psf_func_on_domain(self, energy, pointing_center, domain, lower_radec):
        psf_func = self._get_psf_func(energy, pointing_center, domain, 
                                      lower_radec)
        distances = ((np.arange(ss) - ss//2)*dd for ss,dd in 
                     zip(domain.shape, domain.distances))
        distances = (np.roll(dd, (ss+1)//2) for dd,ss in 
                     zip(distances, domain.shape))
        distances = np.meshgrid(*distances, indexing='ij')
        def func(ra, dec):
            return psf_func(ra, dec, *distances)
        return func
