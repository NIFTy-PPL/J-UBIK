import numpy as np

from ...utils import coord_center


def get_synth_pointsource(info, npix_s, idx_tuple, num_rays):
    """
    Simulate an artificial point source at pixel indices for a specific
    observation.

    Parameters
    ----------
    info: instance of ChandraObersvation
    npix_s : int
        Number of pixels along one spatial axis
    idx_tuple: tuple
        indices of the pointsource. (x_idx, y_idx)
    num_rays: int
        Number of rays for the psf simulation

    Returns
    -------
    nifty8.Field
        with a simulation pointsource at the position idx_tuple
    """
    xy_range = info.obsInfo["xy_range"]
    x_min = info.obsInfo["x_min"]
    y_min = info.obsInfo["y_min"]
    event_f = info.obsInfo["event_file"]
    dy = dx = xy_range * 2 / npix_s
    x_idx, y_idx = idx_tuple
    x_pix_coord = x_min + x_idx * dx
    y_pix_coord = y_min + y_idx * dy
    coords = get_radec_from_xy(x_pix_coord, y_pix_coord, event_f)
    ps = info.get_psf_fromsim(coords, outroot="./psf", num_rays=num_rays)
    return ps


def get_radec_from_xy(temp_x, temp_y, event_f):
    # TODO test precision
    """Calculates sky ra and dec from sky pixel coordinates.

    Parameters
    ----------
    temp_x: int
    temp_y: int

    Returns
    -------
    tuple
    """
    import ciao_contrib.runtool as rt
    rt.dmcoords.punlearn()
    rt.dmcoords(event_f, op="sky", celfmt="deg", x=temp_x, y=temp_y)
    x_p = float(rt.dmcoords.ra)
    y_p = float(rt.dmcoords.dec)
    return (x_p, y_p)


def get_psfpatches(info, n, npix_s, ebin, num_rays=10e6,
                   debug=False, Roll=True, Norm=True):
    """
    Simulating the point spread function of chandra at n**2 positions.
    This is needed for the application of OverlappAdd algorithm at the
    moment.
    # TODO Interpolation of PSF

    Parameters
    -----------

    info: ChandraObservation
    n: int, number of patches along x and y axis
    npix_s: number of pixels along x and y axis
    e_bin: energy bin of info, which is used for the simulation
    num_rays: number of rays for the simulations
    Roll: boolean, if True psf is rolled to the origin.
    Norm: boolean, if True psf is normalized
    debug: boolean, if True: returns also the sources, coordinates(RA/DEC)
    and the positions (indices)

    Returns
    -------
    Array of simulated point spread functions
    """
    xy_range = info.obsInfo["xy_range"]
    x_min = info.obsInfo["x_min"]
    y_min = info.obsInfo["y_min"]
    dy = dx = xy_range * 2 / n
    x_i = x_min + dx * 1 / 2
    y_i = y_min + dy * 1 / 2
    coords = coord_center(npix_s, n)
    psf_sim = []
    source = []
    u = 0
    positions = []
    for i in range(n):
        for l in range(n):
            x_p = x_i + i * dx
            y_p = y_i + l * dy
            radec_c = get_radec_from_xy(x_p, y_p, info.obsInfo["event_file"])
            tmp_psf_sim = info.get_psf_fromsim(radec_c, outroot="./psf",
                                               num_rays=num_rays)
            tmp_psf_sim = tmp_psf_sim[:, :, ebin]
            if Roll:
                tmp_coord = coords[u]
                co_x, co_y = np.unravel_index(tmp_coord, [npix_s, npix_s])
                tmp_psf_sim = np.roll(tmp_psf_sim, (-co_x, -co_y), axis=(0, 1))
                u += 1
            psf_sim.append(tmp_psf_sim)
            if debug:
                tmp_source = np.zeros(tmp_psf_sim.shape)
                pos = np.unravel_index(np.argmax(tmp_psf_sim, axis=None),
                                       tmp_psf_sim.shape)
                tmp_source[pos] = 1
                source.append(tmp_source)
                positions.append(pos)
    psf_sim = np.array(psf_sim)
    if Norm:
        psf_sim = psf_sim / num_rays
    if debug:
        return psf_sim, source, positions, coords
    else:
        return psf_sim
