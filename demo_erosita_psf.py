import xubik0 as xu
from xubik0.operators.erosita_psf import dir_path, fname
import nifty8 as ift
import matplotlib.pyplot as plt

file = dir_path + fname[0]
obs = xu.eROSITA_PSF(file)



energy = '3000'
pointing_center = (250, 250)
lower_radec = (0.,0.)
domain = ift.RGSpace((512,512), distances=(1.,1.))

psf_func = obs.psf_func_on_domain(energy, pointing_center, domain, lower_radec)

tm = psf_func(260., 260.)
plt.imshow(tm.T, origin='lower')
plt.show()

exit()
obs.plot_psfs()