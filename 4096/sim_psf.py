from astropy.io import fits
import numpy as np
import nifty8 as ift
import subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

psf_ra = (3 + 19 / 60 + 48.1 / 3600) * 15
psf_ra = f"ra={psf_ra}"
psf_dec = 41 + 30 / 60 + 42 / 3600
psf_dec = f"dec={psf_dec}"

# psf_ra = (3 + 19 / 60 + 31/ 3600) * 15
# psf_ra = f"ra={psf_ra}"
# psf_dec = 41 + 28 / 60 + 12 / 3600
# psf_dec = f"dec={psf_dec}"
subprocess.run(
    [
        "simulate_psf",
        "infile=data/11713/primary/acisf11713N002_evt2.fits",
        "outroot=test",
        psf_ra,
        psf_dec,
        "monoenergy=2.3",
        "flux=1e-3",
        "spectrum=none",
        "asolfile=data/11713/primary/pcadf375911063N002_asol1.fits",
    ]
)

psf = fits.open("test/psf")["PRIMARY"].data

psf_total = np.zeros([1024] * 2)
psf_total[200:326, 200:326] = psf

fig, ax = plt.subplots()
ax.imshow(psf_total, norm=LogNorm())
fig.savefig("test/test.png", dpi=600)
