import nifty8 as ift
import xubik0 as xu
import numpy as np
import sys
from models import mf_sky, sf_sky
# ############### Config-Utils ###############


def list_from_cfg(value):
    return list(filter(None, (x.strip() for x in value.splitlines())))


class TransitionOperator(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES |self.ADJOINT_TIMES

# ############### sky-Models ##################


def multif_sky(*,
               npix_s,
               npix_e,
               fov,
               e_lim_min,
               e_lim_max,
               spatial_diffuse_asperity_mean,
               spatial_diffuse_asperity_std,
               spatial_diffuse_flexibility_mean,
               spatial_diffuse_flexibility_std,
               spatial_diffuse_fluctuations_mean,
               spatial_diffuse_fluctuations_std,
               spatial_diffuse_loglogavgslope_mean,
               spatial_diffuse_loglogavgslope_std,
               energy_diffuse_asperity_mean,
               energy_diffuse_asperity_std,
               energy_diffuse_flexibility_mean,
               energy_diffuse_flexibility_std,
               energy_diffuse_fluctuations_mean,
               energy_diffuse_fluctuations_std,
               energy_diffuse_loglogavgslope_mean,
               energy_diffuse_loglogavgslope_std,
               diffuse_offset_mean,
               diffuse_offset_std_mean,
               diffuse_offset_std_std,
               spatial_points_loglinear_slope_mean,
               spatial_points_loglinear_slope_std,
               energy_points_asperity_mean,
               energy_points_asperity_std,
               energy_points_flexibility_mean,
               energy_points_flexibility_std,
               energy_points_fluctuations_mean,
               energy_points_fluctuations_std,
               energy_points_loglogavgslope_mean,
               energy_points_loglogavgslope_std,
               points_offset_mean,
               points_offset_std_mean,
               points_offset_std_std,
               points_brightness_alpha,
               points_brightness_q,
               ):
    signal = mf_sky.mf_sky(npix_s,
                    npix_e,
                    fov,
                    e_lim_min,
                    e_lim_max,
                    spatial_diffuse_asperity_mean,
                    spatial_diffuse_asperity_std,
                    spatial_diffuse_flexibility_mean,
                    spatial_diffuse_flexibility_std,
                    spatial_diffuse_fluctuations_mean,
                    spatial_diffuse_fluctuations_std,
                    spatial_diffuse_loglogavgslope_mean,
                    spatial_diffuse_loglogavgslope_std,
                    energy_diffuse_asperity_mean,
                    energy_diffuse_asperity_std,
                    energy_diffuse_flexibility_mean,
                    energy_diffuse_flexibility_std,
                    energy_diffuse_fluctuations_mean,
                    energy_diffuse_fluctuations_std,
                    energy_diffuse_loglogavgslope_mean,
                    energy_diffuse_loglogavgslope_std,
                    diffuse_offset_mean,
                    diffuse_offset_std_mean,
                    diffuse_offset_std_std,
                    spatial_points_loglinear_slope_mean,
                    spatial_points_loglinear_slope_std,
                    energy_points_asperity_mean,
                    energy_points_asperity_std,
                    energy_points_flexibility_mean,
                    energy_points_flexibility_std,
                    energy_points_fluctuations_mean,
                    energy_points_fluctuations_std,
                    energy_points_loglogavgslope_mean,
                    energy_points_loglogavgslope_std,
                    points_offset_mean,
                    points_offset_std_mean,
                    points_offset_std_std,
                    points_brightness_alpha,
                    points_brightness_q,
                    )
    return signal


def singlef_sky(*,
                npix_s,
                fov,
                spatial_diffuse_asperity_mean,
                spatial_diffuse_asperity_std,
                spatial_diffuse_flexibility_mean,
                spatial_diffuse_flexibility_std,
                spatial_diffuse_fluctuations_mean,
                spatial_diffuse_fluctuations_std,
                spatial_diffuse_loglogavgslope_mean,
                spatial_diffuse_loglogavgslope_std,
                diffuse_offset_mean,
                diffuse_offset_std_mean,
                diffuse_offset_std_std,
                points_brightness_alpha,
                points_brightness_q):
    signal = sf_sky.sf_sky(npix_s,
                    fov,
                    spatial_diffuse_asperity_mean,
                    spatial_diffuse_asperity_std,
                    spatial_diffuse_flexibility_mean,
                    spatial_diffuse_flexibility_std,
                    spatial_diffuse_fluctuations_mean,
                    spatial_diffuse_fluctuations_std,
                    spatial_diffuse_loglogavgslope_mean,
                    spatial_diffuse_loglogavgslope_std,
                    diffuse_offset_mean,
                    diffuse_offset_std_mean,
                    diffuse_offset_std_std,
                    points_brightness_alpha,
                    points_brightness_q)
    return signal


def lh(*, sky, energy_bins, dataset_list):
    sky_space = sky.target
    print(sky)
    npix_e = sky_space.shape[2]
    signal_dt = sky.ducktape_left('full_signal')
    signal_fa = ift.FieldAdapter(signal_dt.target['full_signal'], 'full_signal')
    likelihood_list = []

    exp_norm_max, exp_norm_mean, exp_norm_std = xu.get_norm_exposure_patches(cfg['datasets'], position_space, npix_e)
    print(f'Max of exposure-map-norm: {exp_norm_max} \n Mean of exposure-map-norm: {exp_norm_mean} \nStandard deviation of exposure-map-norm: {exp_norm_std}')
    datasets = list_from_cfg(dataset_list)
    for dataset in datasets:
        observation = np.load(dataset, allow_pickle=True).item()

        # PSF
        psf_arr = observation['psf_sim'].val[:, :, energy_bins]
        psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
        psf_field = ift.Field.from_raw(sky_space, psf_arr)
        norm = ift.ScalingOperator(sky_space, psf_field.integrate().val ** -1)
        psf = norm(psf_field)

        # Data
        data = observation["data"].val[:, :, energy_bins]
        data_field = ift.Field.from_raw(sky_space, data)

        # Exp
        exp = observation["exposure"].val[:, :, energy_bins]
        exp_field = ift.Field.from_raw(sky_space, exp)
        normed_exp_field = ift.Field.from_raw(sky_space, exp) * np.mean(exp_norm_mean)
        normed_exposure = ift.makeOp(normed_exp_field)

        # Mask
        mask = xu.get_mask_operator(normed_exp_field)

        # Likelihood
        convolved = xu.convolve_field_operator(psf, signal_fa, space=0)
        signal_response = mask @ normed_exposure @ convolved

        masked_data = mask(data_field)
        likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
        likelihood_list.append(likelihood)

    likelihood_sum = likelihood_list[0]
    for i in range(1, len(likelihood_list)):
        likelihood_sum = likelihood_sum + likelihood_list[i]
    likelihood_sum = likelihood_sum(signal_dt)


def trans(iglobal):
    return None


builder_dct = {"mf_lh": lh, "mf_sky": multif_sky, "trans": trans}


def main():
    _, cfg_file = sys.argv
    cfg = ift.OptimizeKLConfig.from_file(cfg_file, builder_dct)
    cfg.optimize_kl(
        export_operator_outputs={"mf_sky": cfg.instantiate_section("mf_sky")}
    )

if __name__ == "__main__":
    main()