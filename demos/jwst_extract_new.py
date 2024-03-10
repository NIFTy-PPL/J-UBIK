from astropy.wcs import WCS

from jwst import datamodels
import webbpsf
import astropy.units as u
import astropy.coordinates as coord

import numpy as np
import yaml

from sys import exit


def define_location(config):
    from astropy import units
    from astropy.coordinates import SkyCoord

    ra = config['telescope']['pointing']['ra']
    dec = config['telescope']['pointing']['dec']
    frame = config['telescope']['pointing'].get('frame', 'icrs')
    unit = getattr(units, config['telescope']['pointing'].get('unit', 'deg'))

    return SkyCoord(ra=ra*unit, dec=dec*unit, frame=frame)


def get_distances(wcs, loc):
    dpix = wcs.world_to_pixel(loc)
    loc_x = wcs.pixel_to_world(dpix[0]+1, dpix[1])
    loc_y = wcs.pixel_to_world(dpix[0], dpix[1]+1)

    return (loc.separation(loc_x).to(u.arcsec),
            loc.separation(loc_y).to(u.arcsec))


def get_distances_pix(wcs, dpix):
    loc = wcs.pixel_to_world(dpix[0], dpix[1])
    loc_x = wcs.pixel_to_world(dpix[0]+1, dpix[1])
    loc_y = wcs.pixel_to_world(dpix[0], dpix[1]+1)

    return (loc.separation(loc_x).to(u.arcsec),
            loc.separation(loc_y).to(u.arcsec))


config = yaml.load(open('JWST_config.yaml', 'r'), Loader=yaml.SafeLoader)
WORLD_LOCATION = define_location(config)

fov = config['telescope']['fov'] * \
    getattr(u, config['telescope'].get('fov_unit', 'arcsec'))


REFPIX = (256, 12)

test = np.meshgrid(np.arange(0, 512, 1), np.arange(0, 512, 1))

exit()

for ii in range(10):
    REFPIX = np.random.randint(0, 2048, 2)

    xx, yy = [], []
    for ii in range(2):
        filepath = config['files']['filter']['f444w'][ii]
        dm = datamodels.open(filepath)
        header = dm.meta.wcs.to_fits()[0]
        wcs = WCS(header)
        x_dist, y_dist = get_distances_pix(wcs, REFPIX)
        loc = wcs.pixel_to_world(*REFPIX)
        # print(f'Ref={loc.ra, loc.dec}: x_dist = {x_dist}, y_dist = {y_dist}')
        xx.append(x_dist)
        yy.append(y_dist)

    print(f'Diffs {REFPIX}:', xx[0]-xx[1], yy[0]-yy[1])


def create_SkyCoord(ra, dec):
    return coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')


for ii in range(10):
    REFPIX = np.random.randint(0, 2048, 2)

    xx, yy = [], []
    xxa, yya = [], []
    for ii in range(2):
        filepath = config['files']['filter']['f444w'][ii]
        dm = datamodels.open(filepath)

        wtp = dm.meta.wcs.get_transform('world', 'detector')
        ptw = dm.meta.wcs.get_transform('detector', 'world')

        loc = create_SkyCoord(*ptw(*REFPIX))
        loc_x = create_SkyCoord(*ptw(REFPIX[0]+1, REFPIX[1]))
        loc_y = create_SkyCoord(*ptw(REFPIX[0], REFPIX[1]+1))
        x_dist, y_dist = (loc.separation(loc_x).to(u.arcsec),
                          loc.separation(loc_y).to(u.arcsec))
        xx.append(x_dist)
        yy.append(y_dist)

        header = dm.meta.wcs.to_fits()[0]
        wcs = WCS(header)
        x_dist_a, y_dist_a = get_distances_pix(wcs, REFPIX)
        xxa.append(x_dist_a)
        yya.append(y_dist_a)

    # print(f'Diffs {REFPIX} (jwst):', xx[0]-xx[1], yy[0]-yy[1])
    # print(f'Diffs {REFPIX} (astropy):', xxa[0]-xxa[1], yya[0]-yya[1])
    print(f'Diffs {REFPIX} (jwst-astropy) 00:', xx[0]-xxa[0], yy[0]-yya[0])
    print(f'Diffs {REFPIX} (jwst-astropy) 01:', xx[0]-xxa[1], yy[0]-yya[1])
    print(f'Diffs {REFPIX} (jwst-astropy) 10:', xx[1]-xxa[0], yy[1]-yya[0])
    print(f'Diffs {REFPIX} (jwst-astropy) 11:', xx[1]-xxa[1], yy[1]-yya[1])


exit()

for fltname, flt in config['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        dm = datamodels.open(filepath)

        data = dm.data

        header = dm.meta.wcs.to_fits()[0]
        wcs = WCS(header)
        dpix = wcs.world_to_pixel(WORLD_LOCATION)

        x_dist, y_dist = get_distances(
            wcs, coord.SkyCoord(header['CRVAL1']*u.deg,
                                header['CRVAL2']*u.deg, frame='icrs')
        )
        print(f'x_dist = {x_dist}, y_dist = {y_dist}')

        ii0 = int(np.round(d_pix[1])) - FIG_SHAPE//2
        ii1 = int(np.round(d_pix[1])) + FIG_SHAPE//2
        jj0 = int(np.round(d_pix[0])) - FIG_SHAPE//2
        jj1 = int(np.round(d_pix[0])) + FIG_SHAPE//2
