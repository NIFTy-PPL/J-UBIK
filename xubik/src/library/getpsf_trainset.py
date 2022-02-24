import nifty8 as ift
import numpy as np

import matplotlib.pyplot as plt
#from obs.obs4952 import obs4952
#from obs.obs11713 import obs11713
from ..operators.observation_operator import ChandraObservationInformation
#import ciao_contrib.runtool as rt
#from matplotlib.colors import LogNorm

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 1024  # number of spacial bins per axis
npix_e = 4  # number of log-energy bins
fov = 4.0  # FOV in arcmin
elim = (2.0, 10.0)  # energy range in keV
################################################


def plot_single_psf(psf, outname, logscale=True):
    fov = psf.domain[0].distances[0] * psf.domain[0].shape[0] / 2.0
    psf = psf.val.reshape([1024, 1024])
    pltargs = {"origin": "lower", "cmap": "cividis", "extent": [-fov, fov] * 2}
    if logscale == True:
        pltargs["norm"] = LogNorm()
    fig, ax = plt.subplots()
    psf_plot = ax.imshow(psf, **pltargs)
    psf_cbar = fig.colorbar(psf_plot)
    fig.tight_layout()
    fig.savefig(outname, dpi=600)
    plt.close()


outroot = "patches_"
data_domain = ift.DomainTuple.make(
    [
        ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s),
        ift.RGSpace((npix_e,), distances=np.log(elim[1] / elim[0]) / npix_e),
    ]
)
psf_domain = ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s)

info = ChandraObservationInformation(obs4952, npix_s, npix_e, fov, elim, center=None)
ref = (
    info.obsInfo["aim_ra"],
    info.obsInfo["aim_dec"],
)  # Center of the FOV in sky coords

if True:
    info = ChandraObservationInformation(
        obs11713,
        npix_s,
        npix_e,
        fov,
        elim,
        center=(info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]),
    )

n = 8
xy_range = info.obsInfo["xy_range"]
x_min = info.obsInfo["x_min"]
x_max = info.obsInfo["x_max"]
y_min = info.obsInfo["y_min"]
y_max = info.obsInfo["y_max"]
dy = dx = xy_range * 2 / n
x_i = x_min + dx * 1 / 2
y_i = y_min + dy * 1 / 2


def get_radec_from_xy(temp_x, temp_y, event_f):
    rt.dmcoords.punlearn()
    rt.dmcoords(event_f, op="sky", celfmt="deg", x=temp_x, y=temp_y)
    x_p = float(rt.dmcoords.ra)
    y_p = float(rt.dmcoords.dec)
    return (x_p, y_p)


def coord_center(side_length, side_n):
    tdx = tdy = side_length // side_n
    xc = np.arange(tdx // 2, tdx * side_n, tdx)
    yc = np.arange(tdy // 2, tdy * side_n, tdy)
    co = np.array(np.meshgrid(xc, yc)).reshape(2, -1)
    res = np.ravel_multi_index(co, [side_length, side_length])
    return res


coords = coord_center(npix_s, n)
psf_sim = []
source = []
u = 0
for i in range(n):
    for l in range(n):
        x_p = x_i + i * dx
        y_p = y_i + l * dy
        radec_c = get_radec_from_xy(x_p, y_p, info.obsInfo["event_file"])  # PRECISION
        tmp_psf_sim = info.get_psf_fromsim(radec_c, outroot="./psf", num_rays=1e7)
        tmp_psf_sim = tmp_psf_sim[:, :, 0]
        # rollin rollin rollin
        if True:
            tmp_psf_sim = np.roll(tmp_psf_sim, -coords[u])
            u += 1
        psf_field = ift.makeField(psf_domain, tmp_psf_sim)
        norm = ift.ScalingOperator(psf_domain, psf_field.integrate().val ** -1)
        psf_norm = norm(psf_field)
        psf_sim.append(psf_norm)

        tmp_source = np.zeros(tmp_psf_sim.shape)
        pos = np.unravel_index(np.argmax(tmp_psf_sim, axis=None), tmp_psf_sim.shape)
        tmp_source[pos] = 1
        source_field = ift.makeField(psf_domain, tmp_source)
        source.append(source_field)

np.save(outroot + "psf.npy", {"psf_sim": psf_sim, "source": source})
