import xubik0 as xu
import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt

dir_path= "tm1/bcf/"
fname = ["tm1_2dpsf_190219v05.fits", "tm1_2dpsf_190220v03.fits"]

file = dir_path + fname[0]
obs = xu.eROSITA_PSF(file)

energy = '3000'
pointing_center = (250, 250)
lower_radec = (0.,0.)
domain = ift.RGSpace((512,512), distances=(1.,1.))

my_func = obs._get_psf_func(energy, pointing_center, domain, lower_radec)
coords = tuple(np.arange(257) - 128 for _ in range(2))
a = np.where(coords[0] == 0.)[0]
b = np.where(coords[1] == 0.)[0]
print(coords)
coords = np.meshgrid(*coords, indexing='ij')

cc = (250, 250)
im = my_func(cc[0], cc[1], coords[0], coords[1])
plt.imshow(im.T, origin='lower')
plt.scatter(a, b, marker='.', c='r')
plt.show()
exit()

psf_func = obs.psf_func_on_domain(energy, pointing_center, domain, lower_radec)

tm = psf_func(250., 250.)
plt.imshow(tm.T, origin='lower')
plt.show()
exit()
obs.plot_psfs()
