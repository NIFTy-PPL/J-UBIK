import nifty8 as ift
import xubik0 as xu


class ErositaSky:
    def __init__(self, config_file, alpha=None, q=None):
        if not isinstance(config_file, str):
            raise TypeError("The config_file argument needs to be the path to a .yaml config file.")
        # fixme: add all relevant checks and docstrings

        # Prepare the spaces
        self.config = xu.get_cfg(config_file)
        self.priors = self.config['priors']
        self.alpha, self.q = alpha, q

        self.position_space = ift.RGSpace(2 * (self.config['grid']['npix'],))  # FIXME: set right distances
        extended_size = self.config['grid']['padding_ratio'] * self.position_space.shape[0]
        self.extended_space = ift.RGSpace(2 * (extended_size,), distances=self.position_space.distances)

    def create_sky_model(self):
        self.pad = ift.FieldZeroPadder(self.position_space, self.extended_space.shape)
        point_sources = self._create_point_source_model()
        diffuse_component = self._create_diffuse_component_model()
        sky = point_sources + diffuse_component
        return point_sources, diffuse_component, sky

    def _create_point_source_model(self):
        if self.alpha is not None and self.q is not None:
            point_sources = ift.InverseGammaOperator(self.extended_space, alpha=self.alpha, q=self.q)
        else:
            point_sources = ift.InverseGammaOperator(self.extended_space, **self.priors['point_sources'])
        return point_sources.ducktape('point_sources')

    def _create_diffuse_component_model(self):
        cfm = ift.CorrelatedFieldMaker("")
        cfm.set_amplitude_total_offset(**self.priors['diffuse']['offset'])
        cfm.add_fluctuations(self.extended_space, **self.priors['diffuse']['fluctuations'])
        return cfm.finalize().exp()


if __name__ == "__main__":
    config = 'eROSITA_config.yaml'
    model = ErositaSky(config)
    ps, diffuse, sky = model.create_sky_model()

    ift.random.push_sseq_from_seed(model.config['seed'])
    import matplotlib.colors as colors

    n_samples = 6
    ift.plot_priorsamples(ps, n_samples=n_samples, common_colorbar=False, norm=colors.SymLogNorm(linthresh=10e-8), nx=3)
    ift.plot_priorsamples(diffuse, n_samples=n_samples, common_colorbar=False, norm=colors.SymLogNorm(linthresh=10e-8),
                          nx=3)
    ift.plot_priorsamples(sky, n_samples=n_samples, common_colorbar=False, norm=colors.SymLogNorm(linthresh=10e-8),
                          nx=3)
