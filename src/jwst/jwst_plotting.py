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


def build_plot_sky_residuals(
    results_directory: str,
    residual_directory: str,
    data_dict: dict,
    upperleft: tuple[str, jft.Model],
    sky_model_with_key: jft.Model,
    small_sky_model: jft.Model,
    norm: callable,
    sky_min: float = 5e-4,
):

    def sky_residuals(samples: jft.Samples, x: jft.OptimizeVIState):
        print(f"Results: {results_directory}")

        ylen = len(data_dict)
        fig, axes = plt.subplots(ylen, 4, figsize=(12, 3*ylen), dpi=300)
        ims = np.zeros_like(axes)
        for ii, (dkey, data) in enumerate(data_dict.items()):

            dm = data['data_model']
            dd = data['data']
            std = data['std']
            mask = data['mask']
            cm = dm.rotation_and_shift.correction_model

            model_data = []
            for si in samples:
                tmp = np.zeros_like(dd)
                val = sky_model_with_key(si)
                while isinstance(si, jft.Vector):
                    si = si.tree
                val = val | si
                tmp[mask] = dm(val)
                model_data.append(tmp)

            if isinstance(cm, CoordinatesCorrection):
                sh_m, sh_s = jft.mean_and_std(
                    [cm.shift_prior(s) for s in samples])
                ro_m, ro_s = jft.mean_and_std(
                    [cm.rotation_prior(s) for s in samples])
                ro_m, ro_s = ro_m[0], ro_s[0]
                sh_m, sh_s = (sh_m.reshape(2), sh_s.reshape(2))
            else:
                sh_m, sh_s, ro_m, ro_s = (0, 0), (0, 0), 0, 0

            model_mean = jft.mean(model_data)
            redchi_mean, redchi2_std = jft.mean_and_std(
                [redchi2(dd[mask], m[mask], std[mask], dd[mask].size) for m in model_data])

            residual = jft.mean([(dd - md)/std for md in model_data])

            max_d, min_d = np.nanmax(dd), np.nanmin(dd)
            min_d = np.max((min_d, 0.1))
            axes[ii, 1].set_title(f'Data {dkey}')
            ims[ii, 1] = axes[ii, 1].imshow(
                dd, origin='lower', norm=norm(vmin=min_d, vmax=max_d))
            axes[ii, 2].set_title('Data model')
            ims[ii, 2] = axes[ii, 2].imshow(
                model_mean, origin='lower', norm=norm(vmin=min_d, vmax=max_d))
            axes[ii, 3].set_title('Data - Data model')
            ims[ii, 3] = axes[ii, 3].imshow(
                residual, origin='lower', vmin=-3, vmax=3, cmap='RdBu_r')

            data_model_text = '\n'.join(
                (f'dx={sh_m[0]:.1e}+-{sh_s[0]:.1e}',
                 f'dy={sh_m[1]:.1e}+-{sh_s[1]:.1e}',
                 f'dth={ro_m:.1e}+-{ro_s:.1e}')
            )
            chi = '\n'.join((
                f'redChi2: {redchi_mean:.2f} +/- {redchi2_std:.2f}',
            ))
            display_text(axes[ii, 2], data_model_text)
            display_text(axes[ii, 3], chi)

        # Alpha or Mass field
        uleft_name, uleft_model = upperleft
        alpha_mean = jft.mean([uleft_model(si) for si in samples])
        axes[0, 0].set_title(uleft_name)
        ims[0, 0] = axes[0, 0].imshow(alpha_mean, origin='lower')

        # Calculate sky
        mean_small = jft.mean([small_sky_model(si) for si in samples])
        ma, mi = jft.max(mean_small), jft.min(mean_small)
        ma, mi = np.max((sky_min, ma)), np.max((sky_min, mi))

        for ii, filter_name in enumerate(sky_model_with_key.target.keys()):
            axes[ii+1, 0].set_title(f'Sky {filter_name}')
            ims[ii+1, 0] = axes[ii+1, 0].imshow(
                mean_small[ii], origin='lower', norm=norm(vmax=ma, vmin=mi))

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(join(residual_directory, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    return sky_residuals


def build_sky_plot_samples(
    sky_directory: str,
    sky_model: jft.Model,
    small_sky_model: jft.Model,
    sky_model_with_key: jft.Model,
    norm: callable,
    sky_extent: Tuple[int],
):

    def plot_sky_samples(samples: jft.Samples, x: jft.OptimizeVIState):
        ylen, xlen = find_closest_factors(len(samples)+4)

        samps_big = [sky_model(si) for si in samples]
        mean, std = jft.mean_and_std(samps_big)
        mean_small, std_small = jft.mean_and_std(
            [small_sky_model(si) for si in samples])
        flds = [mean_small, std_small/mean_small, mean, std/mean] + samps_big

        for ii, filter_name in enumerate(sky_model_with_key.target.keys()):
            fig, axes = plt.subplots(
                ylen, xlen, figsize=(2*xlen, 1.5*ylen), dpi=300)
            for ax, fld in zip(axes.flatten(), flds):
                im = ax.imshow(
                    fld[ii], origin='lower', norm=norm(), extent=sky_extent)
                fig.colorbar(im, ax=ax, shrink=0.7)
            fig.tight_layout()
            fig.savefig(
                join(sky_directory, f'{x.nit:02d}_{filter_name}.png'), dpi=300)
            plt.close()

    return plot_sky_samples


def build_color_components_plotting(
    sky_model: ColorMixComponents,
    results_directory: str,
    substring='',
):

    colors_directory = join(results_directory, f'colors_{substring}')
    makedirs(colors_directory, exist_ok=True)

    N_comps = len(sky_model.components._comps)

    def cc_print(samples: jft.Samples, x: jft.OptimizeVIState):
        mat_mean, mat_std = jft.mean_and_std(
            [sky_model.color.matrix(si) for si in samples])
        print()
        print('Color Mixing Matrix')
        print(mat_mean, '\n+-\n', mat_std)
        print()

        comps = jft.mean([sky_model.components(si) for si in samples])
        correlated_comps = jft.mean([sky_model(si) for si in samples])

        fig, axes = plt.subplots(2, N_comps, figsize=(4*N_comps, 3))
        for ax, cor_comps, comps in zip(axes.T, correlated_comps, comps):
            im0 = ax[0].imshow(cor_comps, origin='lower', norm=LogNorm())
            im1 = ax[1].imshow(np.exp(comps), origin='lower', norm=LogNorm())
            plt.colorbar(im0, ax=ax[0])
            plt.colorbar(im1, ax=ax[1])
            ax[0].set_title('Correlated Comps')
            ax[1].set_title('Comps')

        plt.tight_layout()
        plt.savefig(join(colors_directory, f'componets_{x.nit}.png'))

    return cc_print


def build_plot(
    data_dict: dict,
    sky_model_with_key: jft.Model,
    sky_model: jft.Model,
    small_sky_model: jft.Model,
    results_directory: str,
    plotting_config: dict,
    upperleft: tuple[str, jft.Model],
):

    residual_directory = join(results_directory, 'residuals')
    sky_directory = join(results_directory, 'sky')
    makedirs(residual_directory, exist_ok=True)

    plot_sky = plotting_config.get('plot_sky', True)
    if plot_sky:
        makedirs(sky_directory, exist_ok=True)

    norm = plotting_config.get('norm', Normalize)
    sky_extent = plotting_config.get('sky_extent', None)

    plot_sky_residuals = build_plot_sky_residuals(
        results_directory=results_directory,
        residual_directory=residual_directory,
        data_dict=data_dict,
        upperleft=upperleft,
        sky_model_with_key=sky_model_with_key,
        small_sky_model=small_sky_model,
        norm=norm)

    plot_sky_samples = build_sky_plot_samples(
        sky_directory=sky_directory,
        sky_model=sky_model,
        small_sky_model=small_sky_model,
        sky_model_with_key=sky_model_with_key,
        norm=norm,
        sky_extent=sky_extent,
    )

    def plot(samples: jft.Samples, x: jft.OptimizeVIState):
        print(f'Plotting: {x.nit}')

        if plot_sky:
            plot_sky_samples(samples, x)

        plot_sky_residuals(samples, x)

    return plot


def plot_sky(sky, data_dict, norm=LogNorm):

    ylen = len(data_dict)
    fig, axes = plt.subplots(ylen, 3, figsize=(9, 3*ylen), dpi=300)
    ims = []
    for ii, (dkey, data) in enumerate(data_dict.items()):

        dm = data['data_model']
        dd = data['data']
        std = data['std']
        mask = data['mask']

        model_data = np.zeros_like(dd)
        model_data[mask] = dm(sky)

        axes[ii, 0].set_title(f'Data {dkey}')
        ims.append(axes[ii, 0].imshow(dd, origin='lower', norm=norm()))
        axes[ii, 1].set_title('Data model')
        ims.append(axes[ii, 1].imshow(
            model_data, origin='lower', norm=norm()))
        axes[ii, 2].set_title('Data - Data model')
        ims.append(axes[ii, 2].imshow((dd - model_data)/std,
                   origin='lower', vmin=-3, vmax=3, cmap='RdBu_r'))

    for ax, im in zip(axes.flatten(), ims):
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.show()


def plot_data_data_model_residuals(
    axes,
    data: ArrayLike,
    data_model: ArrayLike,
    std: ArrayLike,
):
    pass


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
