import nifty8 as ift
import xubik0 as xu
from matplotlib.colors import LogNorm


class SkyModel:
    def __init__(self, config_file):
        if not isinstance(config_file, str):
            raise TypeError("The config_file argument needs to be the path to a .yaml config file.")
        # FIXME: add all relevant checks and docstrings

        # Load config
        self.config = xu.get_cfg(config_file)
        self.priors = self.config['priors']

        # grid info
        grid_info = self.config['grid']
        tel_info = self.config['telescope']

        # Prepare the spaces
        self.position_space = ift.RGSpace(2*(self.config['grid']['npix'],), distances=[tel_info['fov'] / grid_info['npix']])
        extended_size = int(self.config['grid']['padding_ratio'] * self.position_space.shape[0])
        self.extended_space = ift.RGSpace(2*(extended_size,), distances=self.position_space.distances)

        # Prepare zero padding
        self.pad = ift.FieldZeroPadder(self.position_space, self.extended_space.shape)

    def create_sky_model(self):
        diffuse_component, pspec = self._create_diffuse_component_model()
        if self.priors['point_sources'] is None:
            sky = diffuse_component
            sky_dict = {'sky': sky, 'pspec': pspec}
        else:
            point_sources = self._create_point_source_model()
            sky = point_sources + diffuse_component
            sky_dict = {'sky': sky, 'point_sources': point_sources, 'diffuse': diffuse_component,
                        'pspec': pspec}
        return sky_dict

    def _create_point_source_model(self):
        point_sources = ift.InverseGammaOperator(self.extended_space, **self.priors['point_sources'])
        return point_sources.ducktape('point_sources')

    def _create_diffuse_component_model(self):
        # FIXME: externalize power spectrum of diffuse model!
        cfm = ift.CorrelatedFieldMaker("")
        cfm.set_amplitude_total_offset(**self.priors['diffuse']['offset'])
        cfm.add_fluctuations(self.extended_space, **self.priors['diffuse']['fluctuations'])
        diffuse = cfm.finalize().exp()
        pspec = cfm.power_spectrum
        return diffuse, pspec
