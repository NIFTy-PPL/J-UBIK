import nifty8 as ift
import xubik0 as xu
from src.library.erosita_observation import ErositaObservation


class ErositaSky:
    def __init__(self, data, exposure, config_file):
        if not isinstance(config_file, str):
            raise TypeError("The config_file argument needs to be the path to a .yaml config file.")
        #fixme: add all relevant checks and docstrings

        # Prepare the spaces
        self.position_space = ift.RGSpace(data.shape)  # FIXME: set right distances
        self.extended_space = ift.RGSpace((2 * data.shape[0], 2 * data.shape[1]), distances=self.position_space.distances)

        self.data = ift.Field.from_raw(self.position_space, data)  # FIXME: add domain checks and correspondent errors
        self.exposure = ift.Field.from_raw(self.position_space, exposure) if exposure is not None else None
        self.priors = xu.get_cfg(config_file)['priors']
        self.pad = ift.FieldZeroPadder(self.position_space, self.extended_space.shape)

        self.full_sky, self.sky, self.signal, self.mask = self._create_sky_model()
        self.masked_data = self.mask(self.pad(self.data))

    def _create_point_source_model(self):
        point_sources = ift.InverseGammaOperator(self.extended_space, **self.priors['point_sources'])
        return point_sources.ducktape('point_sources')

    def _create_diffuse_component_model(self):
        return ift.SimpleCorrelatedField(self.extended_space, **self.priors['diffuse']).exp()

    def _create_sky_model(self):
        point_sources = self._create_point_source_model()
        diffuse_component = self._create_diffuse_component_model()

        # Correct for exposure
        normed_exposure = xu.get_normed_exposure(self.exposure, self.data) # TODO: check that data is actually 0 where exp is
        padded_normed_exposure = self.pad(normed_exposure)
        exposure_padding = ift.makeOp(padded_normed_exposure)

        mask = xu.get_mask_operator(padded_normed_exposure)

        full_sky = point_sources + diffuse_component
        sky = self.pad.adjoint @ exposure_padding @ full_sky
        signal = mask @ exposure_padding @ full_sky

        # p = ift.Plot()
        # import matplotlib.colors as colors
        # p.add(full_sky(ift.from_random(full_sky.domain)), norm=colors.SymLogNorm(linthresh=10e-1))
        # output_name = "prova_sig_full_sky.png"
        # p.output(name=output_name)
        # print("Output saved as {}.".format(output_name))
        return full_sky, sky, signal, mask


if __name__ == "__main__":
    # Load the data
    obs_path = "../data/"  # Folder that gets mounted to the docker
    filename = "combined_out_08_1_imm.fits"
    input_filename = ['LMC_SN1987A/fm00_700203_020_EventList_c001.fits',
                      'LMC_SN1987A/fm00_700204_020_EventList_c001.fits',
                      'LMC_SN1987A/fm00_700204_020_EventList_c001.fits']

    observation_instance = ErositaObservation(input_filename, filename, obs_path)  # load an observation object
    # observation = observation_instance.get_data(emin=0.7, emax=1.0, image=True, rebin=80, size=3240, pattern=15,
    #                                             gti='GTI') # combine 3 datasets into an image saved in filename

    # observation_instance_2 = ErositaObservation(filename, filename) # load a new observation from the merged image
    # observation_instance_2.get_exposure_maps(filename, 0.7, 1.0, mergedmaps="expmap_combined.fits") # retrieve expmaps

    observation = observation_instance.load_fits_data(filename)
    data = observation[0].data
    image_filename = "combined_out_08_1_imm.png"
    # observation_instance.plot_fits_data(filename, image_filename, slice=(1100, 2000, 800, 2000), dpi=800) # plot data
    data = data[800:2000, 1100:2000]  # slice the data

    # Plot the data check
    # import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    #
    # plt.imshow(data, origin='lower', norm=colors.SymLogNorm(linthresh=10e-1))
    # plt.show()

    expmap = None

    # Load sky model
    erositaModel = ErositaSky(data, expmap, "eROSITA_config.yaml")
    full_sky_field = erositaModel.full_sky(ift.from_random(erositaModel.full_sky.domain))
    sky_field = erositaModel.pad.adjoint(full_sky_field)

    # Plot prior sample
    p = ift.Plot()
    import matplotlib.colors as colors
    p.add(sky_field, norm=colors.SymLogNorm(linthresh=10e-1))
    output_name = "erosita_priors.png"
    p.output(name=output_name)
    print("Output saved as {}.".format(output_name))
