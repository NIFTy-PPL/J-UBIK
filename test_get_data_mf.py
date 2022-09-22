#!/usr/bin/env python3
from astropy.io import fits
import numpy as np

infile = 'data/9107/primary/acisf09107N003_evt2.fits'
with fits.open(infile) as dat_filtered:
    evts = dat_filtered['EVENTS'].data
print(np.log(1.e-3*evts['energy']))
print(np.amin(np.log(1.e-3*evts['energy'])))
print(np.amax(np.log(1.e-3*evts['energy'])))
evts = np.array([evts['x'], evts['y'], np.log(1.e-3*evts['energy'])])
evts = evts.transpose()

print(evts.shape)
bins =(1024, 1024, np.log([0.5, 1.2,2.0,7.0]))
bins_old = (1024, 1024, 2)
ranges_old = (None, None, (np.log(2.0), np.log(10.0)))
ranges = (None, None, (np.log(0.5), np.log(7.0)))
data, edges = np.histogramdd(evts, bins=bins, range =ranges)
data_old, edges_old = np.histogramdd(evts, bins=bins_old, range=ranges_old, normed=False, weights=None)
data = data.transpose((1,0,2)).astype(int)

print(data.shape)
