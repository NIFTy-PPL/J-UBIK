from charm_lensing.lens_system import LensSystem
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np

from os.path import join
from os import makedirs

import nifty8.re as jft

from jubik0.jwst.mock_data.mock_evaluation import redchi2
from jubik0.jwst.mock_data.mock_plotting import display_text
from jubik0.library.sky_colormix import ColorMix
from jubik0.jwst.rotation_and_shift.coordinates_correction import CoordinatesCorrection
from jubik0.jwst.filter_projector import FilterProjector

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
    model: jft.Model,
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


def _get_data_model_and_chi2(
    position_or_samples: Union[dict, jft.Samples],
    sky_or_skies: ArrayLike,
    data_model: jft.Model,
    data: ArrayLike,
    mask: ArrayLike,
    std: ArrayLike,
):
    if isinstance(std, float):
        std = np.full_like(data, std)

    if isinstance(position_or_samples, jft.Samples):
        model_data = []
        for ii, si in enumerate(position_or_samples):
            tmp = np.zeros_like(data)
            while isinstance(si, jft.Vector):
                si = si.tree
            tmp[mask] = data_model(sky_or_skies[ii] | si)
            model_data.append(tmp)
        model_mean = jft.mean(model_data)
        redchi_mean, redchi_std = jft.mean_and_std(
            [redchi2(data[mask], m[mask], std[mask], data[mask].size)
             for m in model_data])

    else:
        model_data = np.zeros_like(data)
        position_or_samples = position_or_samples | sky_or_skies
        model_data[mask] = data_model(position_or_samples)
        redchi_mean = redchi2(
            data[mask], model_data[mask], std[mask], data[mask].size)
        redchi_std = 0
        model_mean = model_data

    return model_mean, (redchi_mean, redchi_std)


def _get_model_samples_or_position(position_or_samples, sky_model):
    if isinstance(position_or_samples, jft.Samples):
        return [sky_model(si) for si in position_or_samples]
    return sky_model(position_or_samples)


def determine_xlen_residuals(data_dict: dict, xmax_residuals):
    maximum = 0
    for dkey in data_dict.keys():
        index = int(dkey.split('_')[-1])
        if index > maximum:
            maximum = index
    maximum += 1
    if maximum > xmax_residuals:
        return xmax_residuals
    return maximum  # because 0 is already there and will not be counted


def determine_xpos(dkey: str):
    index = int(dkey.split('_')[-1])
    return 3 + index


def determine_ypos(dkey: str, filter_projector: FilterProjector):
    ekey = dkey.split('_')[1]
    return filter_projector.keys_and_index[ekey]


def build_plot_sky_residuals(
    results_directory: str,
    filter_projector: FilterProjector,
    data_dict: dict,
    sky_model_with_key: jft.Model,
    small_sky_model: jft.Model,
    overwrite_model: Optional[tuple[tuple[int], str, jft.Model]] = None,
    plotting_config: dict = {},
):
    '''
    overwrite_model:
        - ii, jj
        - name
        - model
    '''

    residual_directory = join(results_directory, 'residuals')
    makedirs(residual_directory, exist_ok=True)

    norm = plotting_config.get('norm', Normalize)
    display_pointing = plotting_config.get('display_pointing', True)
    display_chi2 = plotting_config.get('display_chi2', True)
    std_relative = plotting_config.get('std_relative', True)
    fileformat = plotting_config.get('fileformat', 'png')
    xmax_residuals = plotting_config.get('xmax_residuals', np.inf)

    sky_min = plotting_config.get('sky_min', 5e-4)

    residual_plotting_config = plotting_config.get(
        'data_config', dict(min=5e-4, norm=Normalize))

    ylen = len(sky_model_with_key.target)
    xlen = 3 + determine_xlen_residuals(data_dict, xmax_residuals)

    def sky_residuals(
        position_or_samples: Union[dict, jft.Samples],
        state_or_none: Optional[jft.OptimizeVIState] = None
    ):
        print(f"Results: {results_directory}")
        print('Plotting residuals')

        fig, axes = plt.subplots(ylen, xlen, figsize=(3*xlen, 3*ylen), dpi=300)
        ims = np.zeros_like(axes)
        if ylen == 1:
            ims = ims[None]
            axes = axes[None]

        sky_or_skies = _get_model_samples_or_position(
            position_or_samples, sky_model_with_key)
        for dkey, data in data_dict.items():

            xpos_residual = determine_xpos(dkey)
            ypos = determine_ypos(dkey, filter_projector)
            if xpos_residual > xlen - 1:
                continue

            data_model = data['data_model']
            data_i = data['data']
            std = data['std']
            mask = data['mask']

            model_mean, (redchi_mean, redchi_std) = _get_data_model_and_chi2(
                position_or_samples,
                sky_or_skies,
                data_model=data_model,
                data=data_i,
                mask=mask,
                std=std)

            if xpos_residual == 3:
                ims[ypos, 1:] = _plot_data_data_model_residuals(
                    ims[ypos, 1:],
                    axes[ypos, 1:],
                    data_key=dkey,
                    data=data_i,
                    data_model=model_mean,
                    std=std if std_relative else 1.0,
                    plotting_config=residual_plotting_config)

                if display_pointing:
                    (sh_m, sh_s), (ro_m, ro_s) = get_shift_rotation_correction(
                        position_or_samples,
                        data_model.rotation_and_shift.correction_model)
                    data_model_text = '\n'.join(
                        (f'dx={sh_m[0]:.1e}+-{sh_s[0]:.1e}',
                         f'dy={sh_m[1]:.1e}+-{sh_s[1]:.1e}',
                         f'dth={ro_m:.1e}+-{ro_s:.1e}')
                    )
                    display_text(axes[ypos, 2], data_model_text)

            else:
                ims[ypos, xpos_residual] = axes[ypos, xpos_residual].imshow(
                    (data_i-model_mean)/std, origin='lower',
                    vmin=-3, vmax=3, cmap='RdBu_r')

            if display_chi2:
                chi = '\n'.join((
                    f'redChi2: {redchi_mean:.2f} +/- {redchi_std:.2f}',
                ))
                display_text(axes[ypos, xpos_residual], chi)

        small_mean, small_std = get_position_or_samples_of_model(
            position_or_samples,
            small_sky_model)
        ma, mi = jft.max(small_mean), jft.min(small_mean)
        ma, mi = np.max((sky_min, ma)), np.max((sky_min, mi))
        # FIXME: This should be handled by a source with shape 3
        if len(small_mean.shape) == 2:
            small_mean = small_mean[None]
            small_std = small_std[None]

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
                        f'{state_or_none.nit:02d}.{fileformat}'), dpi=300)
            # plt.clf()
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
    sky_model: ColorMix,
    results_directory: str,
    substring='',
):

    if not hasattr(sky_model, 'components'):
        def _(pos, state=None):
            return None
        return _

    colors_directory = join(
        results_directory, f'colors_{substring}' if substring != '' else 'colors')
    makedirs(colors_directory, exist_ok=True)
    N_comps = len(sky_model.components.components)

    def color_plot(
        position_or_samples: Union[dict, jft.Samples],
        state_or_none: Optional[jft.OptimizeVIState] = None,
    ):
        mat_mean, mat_std = get_position_or_samples_of_model(
            position_or_samples,
            sky_model.color_matrix)
        print()
        print('Color Mixing Matrix')
        print(mat_mean, '\n+-\n', mat_std)
        print()

        components_mean, _ = get_position_or_samples_of_model(
            position_or_samples, sky_model.components)
        correlated_mean, _ = get_position_or_samples_of_model(
            position_or_samples, sky_model)

        fig, axes = plt.subplots(2, N_comps, figsize=(4*N_comps, 6))
        for ax, cor_comps, comps in zip(axes.T, correlated_mean, components_mean):
            im0 = ax[0].imshow(cor_comps, origin='lower', norm=LogNorm())
            im1 = ax[1].imshow(comps, origin='lower')
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
    '''plotting_config:
        - min
        - norm
    '''

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


def get_alpha_nonpar(lens_system: LensSystem):
    ll_nonpar, ll_alpha, sl_nonpar, sl_alpha = [
        lambda _: np.zeros((12, 12)) for ii in range(4)]

    llm, slm = (lens_system.lens_plane_model.light_model.nonparametric(),
                lens_system.source_plane_model.light_model.nonparametric())

    if hasattr(llm, 'alpha'):
        ll_shape = lens_system.lens_plane_model.space.shape
        ll_alpha = jft.Model(
            lambda x: llm.alpha(x)[:ll_shape[0], :ll_shape[1]],
            domain=llm.alpha.domain)

    elif isinstance(llm, jft.CorrelatedMultiFrequencySky):
        call = llm.spectral_index_distribution
        ll_shape = lens_system.lens_plane_model.space.shape
        ll_alpha = jft.Model(
            lambda x: call(x)[:ll_shape[0], :ll_shape[1]],
            domain=llm.domain)

    if hasattr(slm, 'alpha'):
        sl_alpha = slm.alpha
        sl_nonpar = slm.spatial
    elif isinstance(slm, jft.CorrelatedMultiFrequencySky):
        sl_alpha = slm.spectral_index_distribution
        sl_nonpar = slm.reference_frequency_distribution

    try:
        ll_nonpar = lens_system.lens_plane_model.light_model.parametric(
        )[0].nonparametric()
    except:
        pass

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


def build_plot_source(
    results_directory: str,
    plotting_config: dict,
    filter_projector: FilterProjector,
    source_light_model,
    source_light_alpha,
    source_light_parametric,
    source_light_nonparametric,
    attach_name='',
):
    """

    plotting_config:
        norm_source: Normalize (default)
        norm_source_alpha: Normalize (default)
        norm_source_parametric: Normalize (default)
        norm_source_nonparametric: Normalize (default)
        min_source: 1e-5 (default)
        extent: None (default)
    """

    # from charm_lensing.plotting import get_values

    lens_dir = join(results_directory, 'source')
    makedirs(lens_dir, exist_ok=True)

    # plotting_config
    norm_source = plotting_config.get('norm_source', Normalize)
    norm_source_alpha = plotting_config.get('norm_source_alpha', Normalize)
    norm_source_parametric = plotting_config.get(
        'norm_source_parametric', Normalize)
    norm_source_nonparametric = plotting_config.get(
        'norm_source_nonparametric', Normalize)
    min_source = plotting_config.get('min_source', 1e-5)
    extent = plotting_config.get('extent', None)

    freq_len = filter_projector.domain.shape[0]
    xlen = 3
    ylen = 1 + int(np.ceil(freq_len/xlen))

    slight_alpha = source_light_alpha
    slight_parametric = source_light_parametric
    slight_nonparametric = source_light_nonparametric

    def plot_source(
        position_or_samples: Union[jft.Samples, dict],
        state_or_none: Optional[jft.OptimizeVIState] = None,
    ):
        print('Plotting source light')

        if isinstance(position_or_samples, jft.Samples):
            sla = jft.mean([slight_alpha(x) for x in position_or_samples])
            slnonpar = jft.mean(
                [slight_nonparametric(x) for x in position_or_samples])
            slpar = jft.mean(
                [slight_parametric(x) for x in position_or_samples])
            source_light = jft.mean(
                [source_light_model(x) for x in position_or_samples])

        elif isinstance(position_or_samples, dict):
            sla = slight_alpha(position_or_samples)
            slnonpar = slight_nonparametric(position_or_samples)
            slpar = slight_parametric(position_or_samples)
            source_light = source_light_model(position_or_samples)

        fig, axes = plt.subplots(ylen, xlen, figsize=(3*xlen, 3*ylen), dpi=300)
        ims = np.zeros_like(axes)

        # Plot lens light
        axes[0, 0].set_title("Parametric model")
        axes[0, 1].set_title("Nonparametric correction at I0")
        axes[0, 2].set_title("Spectral index")
        ims[0, 0] = axes[0, 0].imshow(
            slpar,
            origin='lower',
            extent=extent,
            norm=norm_source_parametric(vmin=min_source))
        ims[0, 1] = axes[0, 1].imshow(
            slnonpar,
            origin='lower',
            extent=extent, norm=norm_source_nonparametric())
        ims[0, 2] = axes[0, 2].imshow(
            sla, origin='lower', extent=extent, norm=norm_source_alpha())

        axes = axes.flatten()
        ims = ims.flatten()
        for ii, (fltname, fld) in enumerate(
                filter_projector(source_light).items()):
            ii += 3
            axes[ii].set_title(f'{fltname}')
            ims[ii] = axes[ii].imshow(
                fld,
                origin='lower',
                extent=extent,
                norm=norm_source(
                    vmin=np.max((min_source, source_light.min())),
                    vmax=source_light.max()))

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)

        if state_or_none is not None:
            fig.tight_layout()
            fig.savefig(
                join(lens_dir, f'{attach_name}{state_or_none.nit:02d}.png'),
                dpi=300)
            plt.close()
        else:
            plt.show()

    return plot_source


def build_plot_lens_system(
    results_directory: str,
    plotting_config: dict,
    lens_system,  # : LensSystem,
    filter_projector: FilterProjector,
    lens_light_alpha_nonparametric,
    source_light_alpha_nonparametric,
):

    # from charm_lensing.plotting import get_values

    lens_dir = join(results_directory, 'lens')
    makedirs(lens_dir, exist_ok=True)

    # plotting_config
    norm_source = plotting_config.get('norm_source', Normalize)
    norm_source_alpha = plotting_config.get('norm_source_alpha', Normalize)
    norm_source_nonparametric = plotting_config.get(
        'norm_source_nonparametric', Normalize)
    norm_lens = plotting_config.get('norm_lens', Normalize)
    norm_mass = plotting_config.get('norm_mass', Normalize)
    min_source = plotting_config.get('min_source', 1e-5)
    min_lens = plotting_config.get('min_lens', 1e-5)

    tshape = lens_system.get_forward_model_parametric().target.shape
    # FIXME: This should be handled by a source with shape 3
    xlen = tshape[0] + 2 if len(tshape) == 3 else 3

    lens_light_alph, lens_light_nonp = lens_light_alpha_nonparametric
    lens_ext = lens_system.lens_plane_model.space.extent

    slight_alpha, slight_nonparametric = source_light_alpha_nonparametric
    source_ext = lens_system.source_plane_model.space.extend().extent

    def plot_lens_system(
        position_or_samples: Union[jft.Samples, dict],
        state_or_none: Optional[jft.OptimizeVIState],
        parametric: bool,
    ):
        print('Plotting lens system')

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
            sla = jft.mean([slight_alpha(x) for x in position_or_samples])
            slnonpar = jft.mean([slight_nonparametric(x)
                                 for x in position_or_samples])

        elif isinstance(position_or_samples, dict):
            (source_light,
             lens_light,
             lensed_light,
             convergence,
             convergence_nonpar,
             sky) = get_values(position_or_samples)

            lla = lens_light_alph(position_or_samples)
            lln = lens_light_nonp(position_or_samples)
            sla = slight_alpha(position_or_samples)
            slnonpar = slight_nonparametric(position_or_samples)

        # FIXME: This should be handled by a source with shape 3
        if len(lensed_light.shape) == 2:
            lensed_light = lensed_light[None]
        if len(lens_light.shape) == 2:
            lens_light = lens_light[None]
        if len(source_light.shape) == 2:
            source_light = source_light[None]

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
        ims[1, 0] = axes[1, 0].imshow(sla, origin='lower', extent=source_ext,
                                      norm=norm_source_alpha())
        ims[1, 1] = axes[1, 1].imshow(slnonpar, origin='lower', extent=source_ext,
                                      norm=norm_source_nonparametric())

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
                norm=norm_lens(vmin=np.max((min_lens, lens_light.min())),
                               vmax=lens_light.max()))

            axes[1, ii+filter_offset].set_title(f'Source light {filter_name}')
            ims[1, ii+filter_offset] = axes[1, ii+filter_offset].imshow(
                source_light[ii], origin='lower', extent=source_ext,
                norm=norm_source(vmin=np.max((min_source, source_light.min())),
                                 vmax=source_light.max()))

            axes[2, ii+filter_offset].set_title(f'Lensed light {filter_name}')
            ims[2, ii+filter_offset] = axes[2, ii+filter_offset].imshow(
                lensed_light[ii], origin='lower', extent=lens_ext,
                norm=norm_source(vmin=np.max((min_source, lensed_light.min())),
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


def rgb_plotting(
    lens_system: LensSystem,
    samples: jft.Samples,
    three_filter_names: tuple[str] = ('f1000w', 'f770w', 'f560w')
):

    sl = lens_system.source_plane_model.light_model
    ms, mss = jft.mean_and_std([sl(si) for si in samples])

    from astropy import units, cosmology

    sextent = np.array(lens_system.source_plane_model.space.extend().extent)
    extent_kpc = (np.tan(sextent * units.arcsec) *
                  cosmology.Planck13.angular_diameter_distance(4.2).to(
                      units.kpc)
                  ).value

    # f0, f1, f2 = 'f560w', 'f444w', 'f356w'
    # f0, f1, f2 = 'f1000w', 'f770w', 'f560w'
    f0, f1, f2 = three_filter_names

    rgb = np.zeros((384, 384, 3))
    rgb[:, :, 0] = ms[0]
    rgb[:, :, 1] = ms[1]
    rgb[:, :, 2] = ms[2]

    rgb = rgb / np.max(rgb)
    rgb = np.sqrt(rgb)

    from charm_lensing.plotting import display_scalebar, display_text
    import matplotlib.font_manager as fm

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 10))
    ax.imshow(rgb, origin='lower', extent=extent_kpc)
    display_scalebar(ax, dict(size=5, unit='kpc',
                     fontproperties=fm.FontProperties(size=24)))
    display_text(ax,
                 text=dict(s=f0, color='red',
                           fontproperties=fm.FontProperties(size=30)),
                 keyword='top_right',
                 y_offset_ticker=0,)
    display_text(ax,
                 text=dict(s=f1, color='green',
                           fontproperties=fm.FontProperties(size=30)),
                 keyword='top_right',
                 y_offset_ticker=1,)
    display_text(ax,
                 text=dict(s=f2, color='blue',
                           fontproperties=fm.FontProperties(size=30)),
                 keyword='top_right',
                 y_offset_ticker=2,
                 )
    ax.set_xlim(-5.5, 6)
    ax.set_ylim(-4, 6)
    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.savefig(f'{f0}_{f1}_{f2}_source.png')
    plt.close()
