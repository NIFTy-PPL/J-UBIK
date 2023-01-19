import xubik0 as xu
import nifty8 as ift
import matplotlib.pyplot as plt


dir_path= "tm1/bcf/"
fname = ["tm1_2dpsf_190219v05.fits", "tm1_2dpsf_190220v03.fits"]

file = dir_path + fname[0]
obs = xu.eROSITA_PSF(file)

energy = '3000'
pointing_center = (301810.95, -249521.0000000004)
domain = ift.RGSpace((1024,1024), distances=(7.03125, 7.03125))
lower_radec = tuple(cc - ss*dd/2. for cc,ss,dd in 
                    zip(pointing_center, domain.shape, domain.distances))
psf_func = obs.psf_func_on_domain(energy, pointing_center, domain, lower_radec)

center = (ss*dd/2. for ss,dd in zip(domain.shape, domain.distances))
tm = psf_func(*center)
plt.imshow(tm.T, origin='lower')
plt.show()
