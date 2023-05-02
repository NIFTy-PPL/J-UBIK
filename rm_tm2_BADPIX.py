import numpy as np
from astropy.io import fits

if __name__ == '__main__':
    badpix_file = fits.open("tm7_badpix_140602v01.fits")
    badpix = np.vstack(badpix_file[1].data)
    badpix_subselection = badpix[:, :3].astype(np.int32) - 1

    hdulist = fits.open("tm7_detmap_100602v02.fits")
    detmap=hdulist[0].data

    x_fix = badpix_subselection[:, 0]
    y_fix_start = badpix_subselection[:, 1]
    y_fix_end = y_fix_start + badpix_subselection[:, 2] + 1

    for i in range(badpix_subselection[0].shape[0]-1):
        mask = (slice(y_fix_start[i], y_fix_end[i]), x_fix[i])
        detmap[mask] = 0

    import matplotlib.pyplot as plt
    plt.imshow(detmap)
    plt.show()

    hdulist.writeto("newfile_7.fits")
    hdulist.close()
