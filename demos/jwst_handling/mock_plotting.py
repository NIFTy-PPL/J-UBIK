import numpy as np
import nifty8.re as jft
import matplotlib.pyplot as plt

from charm_lensing.analysis_tools import source_distortion_ratio
from scipy.stats import wasserstein_distance
from charm_lensing.plotting import display_text
from charm_lensing.analysis_tools import wmse, redchi2
from nifty8.re.caramel import power_analyze
from os.path import join
from os import makedirs


def find_corners(xy_positions):
    xy_positions = np.array(xy_positions)

    maxx, maxy = np.argmax(xy_positions, axis=1)
    minx, miny = np.argmin(xy_positions, axis=1)

    square_corners = np.array([
        xy_positions[:, maxx],
        xy_positions[:, maxy],
        xy_positions[:, minx],
        xy_positions[:, miny],
    ])
    return square_corners


def point_in_polygon(point, polygon):
    """Determine if the point (x, y) is inside the given polygon.
    polygon is a list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)]"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def pixels_in_rectangle(corners, shape):
    """Get a boolean numpy array where each element indicates whether the
    center of the pixel at that index is inside the rectangle."""

    # Convert corners to a list of tuples
    polygon = [(corners[i][0], corners[i][1]) for i in range(corners.shape[0])]

    pixel_map = np.zeros(shape, dtype=bool)

    min_x = np.floor(np.min(corners[:, 0]))
    max_x = np.ceil(np.max(corners[:, 0]))
    min_y = np.floor(np.min(corners[:, 1]))
    max_y = np.ceil(np.max(corners[:, 1]))

    start_x = max(0, int(min_x))
    end_x = min(shape[1], int(max_x) + 1)
    start_y = max(0, int(min_y))
    end_y = min(shape[0], int(max_y) + 1)

    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            if point_in_polygon((x + 0.5, y + 0.5), polygon):
                pixel_map[x, y] = True

    return pixel_map


def build_evaluation_mask(reco_grid, data_set, comp_sky=None):
    evaluation_mask = np.full(reco_grid.shape, False)

    for data_key in data_set.keys():
        _, data_grid = data_set[data_key]['data'], data_set[data_key]['grid']

        wl_data_centers, _ = data_grid.wcs.wl_pixelcenter_and_edges(
            data_grid.world_extrema)
        px_reco_datapix_cntr = reco_grid.wcs.index_from_wl(wl_data_centers)[0]
        corners = find_corners(px_reco_datapix_cntr.reshape(2, -1))
        tmp_mask = pixels_in_rectangle(corners, reco_grid.shape)
        evaluation_mask += tmp_mask

        if comp_sky is not None:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2)
            ax, ay = axes
            ax.imshow(comp_sky)
            ax.scatter(*corners.T[::-1], color='orange')
            ay.imshow(tmp_mask)
            ay.scatter(*corners.T[::-1], color='orange')
            plt.show()

    return evaluation_mask


def smallest_enclosing_mask(pixel_map):
    """Finds the edge points of the largest and smallest `True` pixels in specified sections of the pixel_map."""
    assert pixel_map.shape[0] == pixel_map.shape[1]

    mask = np.zeros_like(pixel_map)

    shape = pixel_map.shape[0]
    for ii in range(shape):
        if np.all(pixel_map[ii:shape-ii, ii:shape-ii]):
            mask[ii:shape-ii, ii:shape-ii] = True
            return ii, mask

    return ii, mask


def build_plot(
    likelihood_dicts, comparison_sky, sky_model, res_dir, eval_mask
):
    datas = [ll['data'] for ll in likelihood_dicts.values()]
    data_models = [ll['data_model'] for ll in likelihood_dicts.values()]
    masks = [ll['mask'] for ll in likelihood_dicts.values()]
    stds = [ll['std'] for ll in likelihood_dicts.values()]

    out_dir = join(res_dir, 'residuals')
    makedirs(out_dir, exist_ok=True)

    def get_power(field):
        return power_analyze(
            np.fft.fft2(field[smallest_mask].reshape((reshape,)*2)),
            (1.0,)*2  # FAKE DISTANCES ::FIXME::
        )

    def cross_correlation(input, recon):
        return np.fft.ifft2(
            np.fft.fft2(input).conj() * np.fft.fft2(recon)
        ).real.max()

    eval_comp_sky = np.zeros_like(comparison_sky)
    eval_self_sky = np.zeros_like(comparison_sky)

    ii, smallest_mask = smallest_enclosing_mask(eval_mask)
    reshape = eval_mask.shape[0] - 2*ii
    true_power_spectrum = get_power(comparison_sky)

    def plot_pspec(samples, x):
        YLIMS = (1e2, 1e11)

        skys = [sky_model(si) for si in samples]

        pws = [get_power(sky) for sky in skys]
        pw = get_power(jft.mean(skys))

        fig, ax = plt.subplots(1, 1, figsize=(9, 3), dpi=300)
        ax.plot(pw, label='reco', color='blue', linewidth=0.5)
        for pw in pws:
            ax.plot(pw, color='blue', alpha=0.5, linewidth=0.3)
        ax.plot(true_power_spectrum, label='true',
                color='orange', linewidth=0.5)
        plt.ylim(YLIMS)
        ax.loglog()
        ax.legend()
        fig.savefig(join(out_dir, f'pspec_{x.nit:02d}.png'), dpi=300)
        plt.close()

    def plot(samples, x):
        plot_pspec(samples, x)

        sky = jft.mean([sky_model(si) for si in samples])

        eval_comp_sky[eval_mask] = comparison_sky[eval_mask]
        eval_self_sky[eval_mask] = sky[eval_mask]

        vals = dict(
            sdr=source_distortion_ratio(eval_comp_sky, eval_self_sky),
            wd=wasserstein_distance(eval_comp_sky.reshape(-1),
                                    eval_self_sky.reshape(-1)),
            # cc=cross_correlation(eval_comp_sky, eval_self_sky),
        )

        ylen = 1+len(datas)
        fig, axes = plt.subplots(ylen, 3, figsize=(9, 3*ylen), dpi=300)
        ims = []
        for ii, (d, std, dm, mask) in enumerate(zip(datas, stds, data_models, masks)):
            model_data = []
            for si in samples:
                tmp = np.zeros_like(d)
                tmp[mask] = dm(si)
                model_data.append(tmp)

            mod_mean = jft.mean(model_data)
            redchi_mean, redchi2_std = jft.mean_and_std(
                [redchi2(d, m, std, d.size) for m in model_data])

            axes[ii, 0].set_title('Data')
            ims.append(axes[ii, 0].imshow(d, origin='lower'))
            axes[ii, 1].set_title('Data model')
            ims.append(axes[ii, 1].imshow(mod_mean, origin='lower'))
            axes[ii, 2].set_title('Data residual')
            ims.append(axes[ii, 2].imshow((d - mod_mean)/std, origin='lower',
                                          vmin=-3, vmax=3, cmap='RdBu_r'))
            chi = '\n'.join((
                f'MSE/var: {wmse(d, mod_mean, std):.2f}',
                f'redChi2: {redchi_mean:.2f} +/- {redchi2_std:.2f}',
            ))

            display_text(axes[ii, 2], chi)

        axes[ii+1, 0].set_title('Sky')
        ims.append(axes[ii+1, 0].imshow(comparison_sky, origin='lower'))
        axes[ii+1, 1].set_title('Sky model')
        ims.append(axes[ii+1, 1].imshow(sky, origin='lower'))
        axes[ii+1, 2].set_title('Sky residual')
        ims.append(axes[ii+1, 2].imshow(
            (comparison_sky - sky)/comparison_sky, origin='lower',
            vmin=-0.3, vmax=0.3, cmap='RdBu_r'))

        axes[ii+1, 0].contour(eval_mask.T, levels=1, colors='orange')
        axes[ii+1, 1].contour(eval_mask.T, levels=1, colors='orange')
        axes[ii+1, 2].contour(eval_mask.T, levels=1, colors='orange')

        ss = '\n'.join(
            [f'{k}: {v:.3f}' if k != 'cc' else f'{k}: {v:e}' for k, v in vals.items()])
        display_text(axes[ii+1, 2], ss)
        for ax, im in zip(axes.flatten(), ims):
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(join(out_dir, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    return plot
