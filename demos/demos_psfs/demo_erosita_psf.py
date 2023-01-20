import xubik0 as xu
import nifty8 as ift
import matplotlib.pyplot as plt


dir_path= "data/tm1/bcf/"
fname = ["tm1_2dpsf_190219v05.fits", "tm1_2dpsf_190220v03.fits"]

file = dir_path + fname[0]
obs = xu.eROSITA_PSF(file)

energy = '3000'
domain = ift.RGSpace((1024,1024), distances=(7.03125, 7.03125))
pointing_center = tuple(ss*dd/2. for ss,dd in 
                        zip(domain.shape, domain.distances))
psf_func = obs.psf_func_on_domain(energy, pointing_center, domain)

center = (ss*dd/2. for ss,dd in zip(domain.shape, domain.distances))
tm = psf_func(*center)
plt.imshow(tm.T, origin='lower')
plt.show()
