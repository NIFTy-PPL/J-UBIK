from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open("tm2_detmap_100602v02.fits")
detmap=hdulist[0].data


xfix = 215
yfix_1 = slice(165, 384)
yfix_2 = slice(0, 4)
# Not shure if xfix, yfix shouldn't be flipped, depends on convention

detmap[xfix,yfix_1] = 0
detmap[xfix,yfix_2] = 0

hdulist.writeto("newfile.fits")
hdulist.close()
