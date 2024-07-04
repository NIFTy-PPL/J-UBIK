from charm_lensing.build_lens_system import LensSystem
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np

from os.path import join
from os import makedirs

import nifty8.re as jft

from jubik0.jwst.mock_data.mock_evaluation import redchi2
from jubik0.jwst.mock_data.mock_plotting import display_text
from jubik0.library.sky_colormix import ColorMixComponents
from jubik0.jwst.rotation_and_shift.coordinates_correction import CoordinatesCorrection

from typing import Tuple, Union, Optional
from numpy.typing import ArrayLike


def find_closest_factors(number):
    """
    Finds two integers whose multiplication is larger or equal to the input number,
    with the two output numbers being as close together as possible.

    Args:
        number: The input integer number.

    Returns:
        A tuple containing two integers (x, y) such that x * y >= number and the
        difference between x and y is minimized. If no such factors exist, returns
        None.
    """

    # Start with the square root of the number.
    ii = int(np.ceil(number**0.5))

    jj, kk = ii, ii

    jminus = kminus = 0
    while ((jj-jminus)*(kk-kminus) >= number):
        if kminus == jminus:
            jminus += 1
        else:
            kminus += 1

    if jminus == kminus:
        return jj-jminus, kk-kminus+1
    return jj-jminus+1, kk-kminus


def get_position_or_samples_of_model(
    position_or_samples: Union[dict, jft.Samples],
    model: jft.Model
) -> tuple[ArrayLike, ArrayLike]:
    '''Returns (mean, std) of model'''

    if isinstance(position_or_samples, jft.Samples):
        mean, std = jft.mean_and_std(
            [model(si) for si in position_or_samples])
    else:
        mean = model(position_or_samples)
        std = np.full(mean.shape, 0.)

    return mean, std


def get_shift_rotation_correction(
    position_or_samples: Union[dict, jft.Samples],
    correction_model: Optional[CoordinatesCorrection],
):
    if not isinstance(correction_model, CoordinatesCorrection):
        return (0, 0), (0, 0), 0, 0

    shift_mean, shift_std = get_position_or_samples_of_model(
        position_or_samples, correction_model.shift_prior)
    rotation_mean, rotation_std = get_position_or_samples_of_model(
        position_or_samples, correction_model.rotation_prior)
    shift_mean, shift_std = shift_mean.reshape(2), shift_std.reshape(2)
    rotation_mean, rotation_std = rotation_mean[0], rotation_std[0]

    return (shift_mean, shift_std), (rotation_mean, rotation_std)


def get_data_model_chi2(
    position_or_samples: Union[dict, jft.Samples],
    sky_model_with_key: jft.Model,
    data_model: jft.Model,
    data: ArrayLike,
    mask: ArrayLike,
    std: ArrayLike,
):
    if isinstance(std, float):
        std = np.full_like(data, std)

    if isinstance(position_or_samples, jft.Samples):
        model_data = []
        for si in position_or_samples:
            tmp = np.zeros_like(data)
            tmp_sky = sky_model_with_key(si)
            while isinstance(si, jft.Vector):
                si = si.tree
            tmp_sky = tmp_sky | si
            tmp[mask] = data_model(tmp_sky)
            model_data.append(tmp)
        model_mean = jft.mean(model_data)
        redchi_mean, redchi_std = jft.mean_and_std(
            [redchi2(data[mask], m[mask], std[mask], data[mask].size)
             for m in model_data])

    else:
        model_data = np.zeros_like(data)
        tmp_sky = sky_model_with_key(position_or_samples)
        position_or_samples = position_or_samples | tmp_sky
        model_data[mask] = data_model(position_or_samples)
        redchi_mean = redchi2(
            data[mask], model_data[mask], std[mask], data[mask].size)
        redchi_std = 0
        model_mean = model_data

    return model_mean, (redchi_mean, redchi_std)


def build_plot_sky_residuals(
    results_directory: str,
    data_dict: dict,
    sky_model_with_key: jft.Model,
    small_sky_model: jft.Model,
    overwrite_model: Optional[tuple[tuple[int], str, jft.Model]] = None,
    plotting_config: dict = {},
):

    residual_directory = join(results_directory, 'residuals')
    makedirs(residual_directory, exist_ok=True)

    norm = plotting_config.get('norm', Normalize)
    sky_min = plotting_config.get('sky_min', 5e-4)

    residual_plotting_config = plotting_config.get(
        'data_config', dict(min=5e-4, norm=Normalize))

    def sky_residuals(
        position_or_samples: Union[dict, jft.Samples],
        state_or_none: Optional[jft.OptimizeVIState] = None
    ):
        print(f"Results: {results_directory}")

        ylen = len(data_dict)
        fig, axes = plt.subplots(ylen, 4, figsize=(12, 3*ylen), dpi=300)
        ims = np.zeros_like(axes)
        for ii, (dkey, data) in enumerate(data_dict.items()):

            data_model = data['data_model']
            data_i = data['data']
            std = data['std']
            mask = data['mask']

            (sh_m, sh_s), (ro_m, ro_s) = get_shift_rotation_correction(
                position_or_samples,
                data_model.rotation_and_shift.correction_model)
            data_model_text = '\n'.join(
                (f'dx={sh_m[0]:.1e}+-{sh_s[0]:.1e}',
                 f'dy={sh_m[1]:.1e}+-{sh_s[1]:.1e}',
                 f'dth={ro_m:.1e}+-{ro_s:.1e}')
            )

            model_mean, (redchi_mean, redchi_std) = get_data_model_chi2(
                position_or_samples,
                sky_model_with_key=sky_model_with_key,
                data_model=data_model,
                data=data_i,
                mask=mask,
                std=std)
            chi = '\n'.join((
                f'redChi2: {redchi_mean:.2f} +/- {redchi_std:.2f}',
            ))

            ims[ii, 1:] = _plot_data_data_model_residuals(
                ims[ii, 1:],
                axes[ii, 1:],
                data_key=dkey,
                data=data_i,
                data_model=model_mean,
                std=std,
                plotting_config=residual_plotting_config)

            display_text(axes[ii, 2], data_model_text)
            display_text(axes[ii, 3], chi)

        small_mean, small_std = get_position_or_samples_of_model(
            position_or_samples,
            small_sky_model)
        ma, mi = jft.max(small_mean), jft.min(small_mean)
        ma, mi = np.max((sky_min, ma)), np.max((sky_min, mi))

        for ii, filter_name in enumerate(sky_model_with_key.target.keys()):
            axes[ii, 0].set_title(f'Sky {filter_name}')
            ims[ii, 0] = axes[ii, 0].imshow(
                small_mean[ii], origin='lower', norm=norm(vmax=ma, vmin=mi))

        for kk, (jj, filter_name) in zip(
                range(ii+1, len(axes)),
                enumerate(sky_model_with_key.target.keys())):
            axes[kk, 0].set_title(f'Sky {filter_name} std')
            ims[kk, 0] = axes[kk, 0].imshow(
                small_std[jj], origin='lower', norm=norm(vmax=ma, vmin=mi))

        if overwrite_model is not None:
            ii, jj = overwrite_model[0]
            name = overwrite_model[1]
            model = overwrite_model[2]
            mean, _ = get_position_or_samples_of_model(
                position_or_samples, model)
            axes[ii, jj].set_title(name)
            ims[ii, jj] = axes[ii, jj].imshow(mean, origin='lower')

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()

        if state_or_none is None:
            plt.show()
        else:
            fig.savefig(join(residual_directory,
                        f'{state_or_none.nit:02d}.png'), dpi=300)
            plt.close()

    return sky_residuals


def build_plot_model_samples(
    results_directory: str,
    model_name: str,
    model: jft.Model,
    mapping_axis: Optional[int] = None,
    plotting_config: dict = {},
):
    sky_directory = join(results_directory, model_name)
    makedirs(sky_directory, exist_ok=True)

    norm = plotting_config.get('norm', Normalize)
    sky_min = plotting_config.get('min', 5e-4)
    extent = plotting_config.get('extent')

    def plot_sky_samples(samples: jft.Samples, x: jft.OptimizeVIState):
        samps_big = [model(si) for si in samples]
        mean, std = jft.mean_and_std(samps_big)
        vmin = np.max((mean.min(), sky_min))
        vmax = mean.max()

        if mapping_axis is None:
            ylen, xlen = find_closest_factors(len(samples)+2)
        else:
            ylen, xlen = model.target[mapping_axis], len(samples)+2
        fig, axes = plt.subplots(
            ylen, xlen, figsize=(2*xlen, 1.5*ylen), dpi=300)

        if mapping_axis is None:
            axes = [axes]

        for axi in axes:
            for ax, fld in zip(axi.flatten(), samps_big):
                im = ax.imshow(fld, origin='lower', extent=extent,
                               norm=norm(vmin=vmin, vmax=vmax))
                fig.colorbar(im, ax=ax, shrink=0.7)

        fig.tight_layout()
        fig.savefig(join(sky_directory, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    return plot_sky_samples


def build_color_components_plotting(
    sky_model: ColorMixComponents,
    results_directory: str,
    substring='',
):

    if not hasattr(sky_model, 'components'):
        def _(pos, state=None):
            return None
        return _

    colors_directory = join(results_directory, f'colors_{substring}')
    makedirs(colors_directory, exist_ok=True)
    N_comps = len(sky_model.components._comps)

    def color_plot(
        position_or_samples: Union[dict, jft.Samples],
        state_or_none: Optional[jft.OptimizeVIState] = None,
    ):
        mat_mean, mat_std = get_position_or_samples_of_model(
            position_or_samples,
            sky_model.color.matrix)
        print()
        print('Color Mixing Matrix')
        print(mat_mean, '\n+-\n', mat_std)
        print()

        components_mean, _ = get_position_or_samples_of_model(
            position_or_samples, sky_model.components)
        correlated_mean, _ = get_position_or_samples_of_model(
            position_or_samples, sky_model)

        fig, axes = plt.subplots(2, N_comps, figsize=(4*N_comps, 3))
        for ax, cor_comps, comps in zip(axes.T, correlated_mean, components_mean):
            im0 = ax[0].imshow(cor_comps, origin='lower', norm=LogNorm())
            im1 = ax[1].imshow(np.exp(comps), origin='lower', norm=LogNorm())
            plt.colorbar(im0, ax=ax[0])
            plt.colorbar(im1, ax=ax[1])
            ax[0].set_title('Correlated Comps')
            ax[1].set_title('Comps')

        plt.tight_layout()
        if isinstance(state_or_none, jft.OptimizeVIState):
            plt.savefig(
                join(colors_directory, f'componets_{state_or_none.nit}.png'))
            plt.close()
        else:
            plt.show()

    return color_plot


def _plot_data_data_model_residuals(
    ims: list,
    axes: list,
    data_key: str,
    data: ArrayLike,
    data_model: ArrayLike,
    std: ArrayLike,
    plotting_config: dict = {}
):

    min = plotting_config.get('min', 5e-4)
    norm = plotting_config.get('norm', Normalize)

    max_d, min_d = np.nanmax(data), np.max((np.nanmin(data), min))

    axes[0].set_title(f'Data {data_key}')
    axes[1].set_title('Data model')
    axes[2].set_title('Data - Data model')
    ims[0] = axes[0].imshow(data, origin='lower',
                            norm=norm(vmin=min_d, vmax=max_d))
    ims[1] = axes[1].imshow(data_model, origin='lower',
                            norm=norm(vmin=min_d, vmax=max_d))
    ims[2] = axes[2].imshow((data-data_model)/std, origin='lower',
                            vmin=-3, vmax=3, cmap='RdBu_r')

    return ims


def get_alpha_nonpar(lens_system, plot_components_switch):
    if hasattr(lens_system.lens_plane_model.light_model.nonparametric(), '_sky_model'):
        tmp_alpha = lens_system.lens_plane_model.light_model.nonparametric()._sky_model.alpha_cf
        ll_shape = lens_system.lens_plane_model.space.shape
        ll_alpha = jft.Model(
            lambda x: tmp_alpha(x)[:ll_shape[0], :ll_shape[1]],
            domain=tmp_alpha.domain)
    else:
        def ll_alpha(_): return np.zeros((12, 12))

    try:
        ll_nonpar = lens_system.lens_plane_model.light_model.parametric(
        ).nonparametric()[0].nonparametric()
    except IndexError:
        ll_nonpar = None
    if ll_nonpar is None:
        def ll_nonpar(_): return np.zeros((12, 12))

    if plot_components_switch:

        def sl_nonpar(_): return np.zeros((12, 12))
        def sl_alpha(_): return np.zeros((12, 12))

    else:
        sl_alpha = lens_system.source_plane_model.light_model.nonparametric()._sky_model.alpha_cf
        sl_nonpar = lens_system.source_plane_model.light_model.nonparametric()._sky_model.spatial_cf
        if sl_nonpar is None:
            def sl_nonpar(_): return np.zeros((12, 12))

    return ll_alpha, ll_nonpar, sl_alpha, sl_nonpar


def build_get_values(
    lens_system: LensSystem,
    parametric: bool,
):
    source_model = lens_system.source_plane_model.light_model

    if parametric:
        convergence_model = lens_system.lens_plane_model.convergence_model.parametric()
        def convergence_nonpar_model(_): return np.zeros((12, 12))

        sky_model = lens_system.get_forward_model_parametric()
        lensed_light_model = lens_system.get_forward_model_parametric(
            only_source=True)

    else:
        convergence_model = lens_system.lens_plane_model.convergence_model
        convergence_nonpar_model = lens_system.lens_plane_model.convergence_model.nonparametric()

        sky_model = lens_system.get_forward_model_full()
        lensed_light_model = lens_system.get_forward_model_full(
            only_source=True)

    lens_light_model = lens_system.get_lens_light()
    if lens_light_model is None:
        def lens_light_model(_): return np.zeros((12, 12))
    else:
        lens_light_model = lens_light_model

    def get_values(position):
        return (source_model(position),
                lens_light_model(position),
                lensed_light_model(position),
                convergence_model(position),
                convergence_nonpar_model(position),
                sky_model(position))

    return get_values


def build_plot_lens_system(
    results_directory: str,
    plotting_config: dict,
    lens_system,  # : LensSystem,
    filter_projector,
    lens_light_alpha_nonparametric,
    source_light_alpha_nonparametric,
):

    # from charm_lensing.plotting import get_values

    lens_dir = join(results_directory, 'lens')
    makedirs(lens_dir, exist_ok=True)

    norm_source = plotting_config.get('norm_source', Normalize)
    norm_lens = plotting_config.get('norm_lens', Normalize)
    norm_mass = plotting_config.get('norm_mass', Normalize)

    xlen = len(lens_system.get_forward_model_parametric().target) + 2

    lens_light_alph, lens_light_nonp = lens_light_alpha_nonparametric
    lens_ext = lens_system.lens_plane_model.space.extent

    source_light_alph, source_light_nonp = source_light_alpha_nonparametric
    source_ext = lens_system.source_plane_model.space.extend().extent

    def plot_lens_system(
        position_or_samples: Union[jft.Samples, dict],
        state_or_none: Optional[jft.OptimizeVIState],
        parametric: bool,
    ):

        get_values = build_get_values(lens_system, parametric)

        if isinstance(position_or_samples, jft.Samples):
            (source_light,
             lens_light,
             lensed_light,
             convergence,
             convergence_nonpar,
             sky) = jft.mean([get_values(si) for si in position_or_samples])

            lla = jft.mean([lens_light_alph(x) for x in position_or_samples])
            lln = jft.mean([lens_light_nonp(x) for x in position_or_samples])
            sla = jft.mean([source_light_alph(x) for x in position_or_samples])
            sln = jft.mean([source_light_nonp(x) for x in position_or_samples])

        elif isinstance(position_or_samples, dict):
            (source_light,
             lens_light,
             lensed_light,
             convergence,
             convergence_nonpar,
             sky) = get_values(position_or_samples)

            lla = lens_light_alph(position_or_samples)
            lln = lens_light_nonp(position_or_samples)
            sla = source_light_alph(position_or_samples)
            sln = source_light_nonp(position_or_samples)

        fig, axes = plt.subplots(3, xlen, figsize=(3*xlen, 8), dpi=300)
        ims = np.zeros_like(axes)
        filter_offset = 2

        # Plot lens light
        axes[0, 0].set_title("Lens light alpha")
        axes[0, 1].set_title("Lens light nonpar")
        ims[0, 0] = axes[0, 0].imshow(lla, origin='lower')
        ims[0, 1] = axes[0, 1].imshow(lln, origin='lower')

        # Plot source light
        axes[1, 0].set_title("Source light alpha")
        axes[1, 1].set_title("Source light nonpar")
        ims[1, 0] = axes[1, 0].imshow(sla, origin='lower', extent=source_ext)
        ims[1, 1] = axes[1, 1].imshow(sln, origin='lower', extent=source_ext)

        # Plot mass field
        axes[2, 0].set_title("Mass par")
        axes[2, 1].set_title("Mass nonpar")
        ims[2, 0] = axes[2, 0].imshow(
            convergence, origin='lower', norm=LogNorm())
        ims[2, 1] = axes[2, 1].imshow(
            convergence_nonpar, origin='lower', norm=norm_mass())

        for ii, filter_name in enumerate(filter_projector.target.keys()):
            axes[0, ii+filter_offset].set_title(f'Lens light {filter_name}')
            ims[0, ii+filter_offset] = axes[0, ii+filter_offset].imshow(
                lens_light[ii], origin='lower', extent=lens_ext,
                norm=norm_lens(vmin=np.max((1e-5, lens_light.min())),
                               vmax=lens_light.max()))

            axes[1, ii+filter_offset].set_title(f'Source light {filter_name}')
            ims[1, ii+filter_offset] = axes[1, ii+filter_offset].imshow(
                source_light[ii], origin='lower', extent=source_ext,
                norm=norm_source(vmin=np.max((1e-5, source_light.min())),
                                 vmax=source_light.max()))

            axes[2, ii+filter_offset].set_title(f'Lensed light {filter_name}')
            ims[2, ii+filter_offset] = axes[2, ii+filter_offset].imshow(
                lensed_light[ii], origin='lower', extent=lens_ext,
                norm=norm_source(vmin=np.max((1e-5, lensed_light.min())),
                                 vmax=lensed_light.max()))

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()

        if state_or_none is not None:
            fig.savefig(
                join(lens_dir, f'{state_or_none.nit:02d}.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    return plot_lens_system
