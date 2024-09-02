import numpy as np

from ducc0.fft import good_size as good_fft_size
import nifty8.re as jft
import jubik0 as ju
from jax import numpy as jnp


def fuse_model_components(model_a, model_b):
    """ Takes two models A and B and fuses them to a model C such that the application
    of model C to some latent space x yields C(X) = A(X) + B(X)

    Parameters
    ----------
    model_a: jft.Model
        Model for a sky component A
    model_b: jft.Model
        Model for a sky component B
    Returns
    -------
    model_c: jft.Model
        Model for a sky component C
    """
    fusion = lambda x: model_a(x) + model_b(x)
    domain = model_a.domain
    domain.update(model_b.domain)
    return jft.Model(fusion, domain=domain)


class SkyModel:
    """
    Basic spatial SkyModel consisting of a diffuse (correlated) component
    and a point-source like (not correlated) component. The latter can
    be switched off.

    Parameters:
    ----------
    config_file : python-dictionary, containing information about the grid,
                   (the telescope (for the pixelization)), # TODO IMHO this should be part of the grid
                   the priors on the correlated field and the uncorrelated inverse gamma
                    component.
    """

    def __init__(self, config_file_path=None):

        """Gets the parameters needed for building the sky model from the config file
        given the corresponding path.

        Parameters
        ----------
        config_file_path : string
            Path to the config file
        """
        if config_file_path is not None:
            if not isinstance(config_file_path, str):
                raise TypeError("The path to the config file needs to be a string")
            if not config_file_path.endswith('.yaml') and not config_file_path.endswith('.yml'):
                raise ValueError("The sky model parameters need to be saved in a .yaml or .yml "
                                 "file.")

            self.config = ju.get_config(config_file_path)
        else:
            self.config = {}
        self.s_distances = None
        self.e_distances = None
        self.diffuse = None
        self.point_sources = None
        self.spatial_pspec = None
        self.sky = None

        self.plaw = None
        self.dev_cf = None
        self.dev_pspec = None
        self.alpha_cf = None
        self.alpa_pspec = None

        self.points_plaw = None
        self.points_dev_cf = None
        self.points_dev_pspec = None
        self.points_alpha_cf = None
        self.points_alpha_pspec = None

    def create_sky_model(self, sdim=None, edim=None, s_padding_ratio=None,
                         e_padding_ratio=None, fov=None, e_min=None, e_max=None, e_ref=None,
                         priors=None):
        """Returns the sky model composed out of components given the grid information
        (# pixels and padding ration), the telescope information (FOV: field of view) as well as a
        dictionary for the prior parameters. All these parameters can be set externally or taken
        the SkyModels config file, if they are set to None.

        Parameters
        ----------
        sdim: int or tuple of int
            Number of pixels in each spatial dimension
        edim: int
            Number of pixels in spectral direction
        s_padding_ratio: float
            Ratio between number of pixels in the actual space and the padded space
        e_padding_ratio: float
            Ratio between number of pixels in the actual enegery space and the padded energy space.
            It needs to be taken such that the correlated fields in energy direction has more than 3
            pixels.
        fov: float
            FOV of the telescope
        energy_range:
            Total range of energies (i.e. max. - min. energy)
        priors: dict
            Dictionary of prior parameters for the correlated field
            in the format:
                    point_sources:
                        spatial:
                            alpha:
                            q:
                        plaw: optional
                        dev: optional
                    diffuse:
                        spatial:
                            offset:
                                offset_mean:
                                offset_std:
                            fluctuations:
                                fluctuations:
                                loglogavgslope:
                                flexibility:
                                asperity:
                                harmonic_type:
                                non_parametric_kind:
                            prefix:
                        plaw: optional
                        dev: optional

        Returns
        -------
        sky: jft.Model
        """
        if 'grid' not in self.config.keys():
            self.config['grid'] = {}
        if 'telescope' not in self.config.keys():
            self.config['telescope'] = {}
        if sdim is None:
            sdim = self.config['grid']['sdim']
        else:
            self.config['grid']['sdim'] = sdim
        if edim is None:
            edim = self.config['grid']['edim']
        else:
            self.config['grid']['edim'] = edim
        if s_padding_ratio is None:
            s_padding_ratio = self.config['grid']['s_padding_ratio']
        else:
            self.config['grid']['s_padding_ratio'] = s_padding_ratio
        if e_padding_ratio is None:
            e_padding_ratio = self.config['grid']['e_padding_ratio']
        else:
            self.config['grid']['e_padding_ratio'] = e_padding_ratio
        if fov is None:
            fov = self.config['telescope']['fov']
        else:
            self.config['telescope']['fov'] = fov
        if 'energy_bin' not in self.config['grid'].keys():
            self.config['grid']['energy_bin'] = {}
        if e_min is None:
            e_min = self.config['grid']['energy_bin']['e_min']
        else:
            self.config['grid']['energy_bin']['e_min'] = e_min
        if e_max is None:
            e_max = self.config['grid']['energy_bin']['e_max']
        else:
            self.config['grid']['energy_bin']['e_max'] = e_max
        if e_ref is not None:
            self.config['grid']['energy_bin']['e_ref'] = e_ref
        if priors is None:
            priors = self.config['priors']
        else:
            self.config['priors'] = priors

        sdim = 2 * (sdim,)
        self.s_distances = fov / sdim[0]
        energy_range = np.array(e_max) - np.array(e_min)
        self.e_distances = energy_range / edim # FIXME: add proper distances for irregular energy grid

        if not isinstance(self.e_distances, float) and 'dev_corr' in priors['diffuse'].keys():
            raise ValueError('Grid distances in energy direction have to be regular and defined by'
                             'one float of a corrlated field in energy direction is taken.')

        self._create_diffuse_component_model(sdim, edim, s_padding_ratio, e_padding_ratio,
                                             self.s_distances, self.e_distances, priors['diffuse'])
        if 'point_sources' not in priors:
            self.sky = self.diffuse
        else:
            if 'dev_corr' in priors['point_sources'].keys():
                raise ValueError('Grid distances in energy direction have to be regular and defined by'
                             'one float of a corrlated field in energy direction is taken.')
            self._create_point_source_model(sdim, edim, e_padding_ratio,
                                            self.e_distances, priors['point_sources'])
            self.sky = fuse_model_components(self.diffuse, self.point_sources)
        return self.sky

    def _create_correlated_field(self, shape, distances, prior_dict):
        """ Returns a 1- or 2-dim correlated field and the corresponding
        power spectrum given its shape, the distances and the prior specification.

        Parameters
        ----------
        shape: int or tuple of int
            Number of pixels of the correlated field in one or in two dimensions
        distances : tuple of float or float
            Distances in the space of the correlated field
        prior_dict: dict
            Dictionary of prior parameters for the correlated field
            in the format:
                    offset:
                        offset_mean:
                        offset_std:
                    fluctuations:
                        fluctuations:
                        loglogavgslope:
                        flexibility:
                        asperity:
                        harmonic_type:
                        non_parametric_kind:
                    prefix:

        Returns
        -------
        cf: jft.Model
            Model for the correlated field
        pspec: Callable
            Power spectrum of the correlated field
        """

        cfm = jft.CorrelatedFieldMaker(prefix=prior_dict['prefix'])
        cfm.set_amplitude_total_offset(**prior_dict['offset'])
        cfm.add_fluctuations(shape, distances, **prior_dict['fluctuations'])
        cf = cfm.finalize()
        return cf, cfm.power_spectrum

    def _create_wiener_process(self, x0, sigma, dE, name, edims):
        return jft.WienerProcess(x0, tuple(sigma), dt=dE, name=name, N_steps=edims - 1)

    def _create_diffuse_component_model(self, sdim, edim, s_padding_ratio, e_padding_ratio,
                                        sdistances, edistances, prior_dict):
        """ Returns a model for the diffuse component given the information on its shape and
        distances and the prior dictionaries for the offset and the fluctuations

        Parameters

        ----------
        sdim: int or tuple of int
            Number of pixels in each spatial dimension
        edim: int
            Number of pixels in spectral direction
        s_padding_ratio: float
            Ratio between number of pixels in the actual space and the padded space
        e_padding_ratio: float
            Ratio between number of pixels in the actual enegery space and the padded energy space.
            It needs to be taken such that the correlated fields in energy direction has more than 3
            pixels.
        sdistances : tuple of float or float
            Position-space distances
        edistances : tuple of float or float
            Energy-space distances
        prior_dict: dict
            Dictionary of prior parameters for the correlated field
            in the format:
                    spatial:
                        offset:
                            offset_mean:
                            offset_std:
                        fluctuations:
                            fluctuations:
                            loglogavgslope:
                            flexibility:
                            asperity:
                            harmonic_type:
                            non_parametric_kind:
                        prefix:
                    plaw: optional
                    dev: optional

        Returns
        -------
        diffuse: jft.Model
            Model for the diffuse component
        """
        if not 'spatial' in prior_dict:
            return ValueError('Every diffuse component needs a spatial component')
        if 'dev_wp' in prior_dict and 'dev_corr' in prior_dict:
            raise ValueError('You can only inlude Wiener process or correlated field'
                             'for the deviations around the plaw.')
        ext_s_shp = tuple(good_fft_size(int(entry * s_padding_ratio))
                          for entry in sdim)
        ext_e_shp = good_fft_size(int(edim * e_padding_ratio))
        self.spatial_cf, self.spatial_pspec = self._create_correlated_field(ext_s_shp,
                                                                            sdistances,
                                                                            prior_dict['spatial'])
        if 'plaw' in prior_dict:
            self.alpha_cf, self.alpa_pspec = self._create_correlated_field(ext_s_shp,
                                                                           sdistances,
                                                                           prior_dict['plaw'])
            self.plaw = ju.build_power_law(self._log_rel_ebin_centers(), self.alpha_cf)

        if 'dev_corr' in prior_dict:
            dev_cf, self.dev_pspec = self._create_correlated_field(ext_e_shp,
                                                                   edistances,
                                                                   prior_dict['dev_cor'])
            self.dev_cf = ju.MappedModel(dev_cf, prior_dict['dev_corr']['prefix']+'xi',
                                         ext_s_shp, False)
        if 'dev_wp' in prior_dict:
            dev_cf = self._create_wiener_process(edims=len(self._log_rel_ebin_centers()),
                                                 dE=self._log_dE(),
                                                 **prior_dict['dev_wp'])
            self.dev_cf = ju.MappedModel(dev_cf, prior_dict['dev_wp']['name'],
                                         ext_s_shp, False)
        log_diffuse = ju.GeneralModel({'spatial': self.spatial_cf,
                                       'freq_plaw': self.plaw,
                                       'freq_dev': self.dev_cf}).build_model()
        exp_padding = lambda x: jnp.exp(log_diffuse(x)[:edim, :sdim[0], :sdim[1]])
        self.diffuse = jft.Model(exp_padding, domain=log_diffuse.domain)

    def _create_point_source_model(self, sdim, edim, e_padding_ratio, edistances, prior_dict):
        """ Returns a model for the point-source component given the information on its shape
         and information on the shape and scaling parameters

        Parameters
        ----------
        sdim: int or tuple of int
            Number of pixels in each spatial dimension
        edim: int
            Number of pixels in spectral direction
        e_padding_ratio: float
            Ratio between number of pixels in the actual enegery space and the padded energy space.
            It needs to be taken such that the correlated fields in energy direction has more than 3
            pixels.
        edistances : tuple of float or float
            Energy-space distances
        prior_dict: dict
            Dictionary of prior parameters for the correlated field
            in the format:
                    spatial:
                        alpha:
                        q:
                    plaw: optional
                    dev: optional

        Returns
        -------
        points: jft.Model
            Model for the point-source component
        """
        if 'spatial' not in prior_dict:
            return ValueError('Point source component needs a spatial component')
        if 'dev_wp' in prior_dict and 'dev_corr' in prior_dict:
            raise ValueError('You can only inlude Wiener process or correlated field'
                             'for the deviations around the plaw.')
        ext_e_shp = int(edim * e_padding_ratio)
        point_sources = jft.invgamma_prior(a=prior_dict['spatial']['alpha'],
                                           scale=prior_dict['spatial']['q'])
        points_log_func = lambda x: jnp.log(point_sources(x[prior_dict['spatial']['key']]))
        self.points_log_invg = jft.Model(points_log_func,
                                     domain={prior_dict['spatial']['key']: jft.ShapeWithDtype(sdim)})

        if 'plaw' in prior_dict:
            self.points_alpha = jft.NormalPrior(prior_dict['plaw']['mean'], prior_dict['plaw']['std'],
                                               name=prior_dict['plaw']['name'],
                                               shape=sdim, dtype=jnp.float64)
            points_plaw = ju.build_power_law(self._log_rel_ebin_centers(), self.points_alpha)
            self.points_plaw = jft.Model(lambda x: points_plaw(x),
                                    domain=points_plaw.domain)

        if 'dev_corr' in prior_dict:
            points_dev_cf, self.points_dev_pspec = self._create_correlated_field(ext_e_shp,
                                                                   edistances,
                                                                   prior_dict['dev_cor'])
            self.points_dev_cf = ju.MappedModel(points_dev_cf, prior_dict['dev_corr']['prefix']+'xi',
                                         sdim, False)
        if 'dev_wp' in prior_dict:
            points_dev_cf = self._create_wiener_process(edims=len(self._log_rel_ebin_centers()),
                                                        dE=self._log_dE(),
                                                        **prior_dict['dev_wp'])

            points_dev_cf = ju.MappedModel(points_dev_cf, prior_dict['dev_wp']['name'],
                                                sdim, False)

            self.points_dev_cf = jft.Model(lambda x: points_dev_cf(x),
                                      domain=points_dev_cf.domain)

        log_points = ju.GeneralModel({'spatial': self.points_log_invg,
                                      'freq_plaw': self.points_plaw,
                                      'freq_dev': self.points_dev_cf}).build_model()

        exp_padding = lambda x: jnp.exp(log_points(x)[:edim, :, :])
        self.point_sources = jft.Model(exp_padding, domain=log_points.domain)

    def sky_model_to_dict(self):
        """Return a dictionary with callables for the major sky models."""
        sky_dict = {'sky': self.sky,
                    'diffuse': self.diffuse,
                    'points': self.point_sources}
        no_none_dict = {key: value for (key, value) in sky_dict.items() if value is not None}
        return no_none_dict

    def _log_ebin_centers(self):
        e_max = np.array(self.config['grid']['energy_bin']['e_max'])
        e_min = np.array(self.config['grid']['energy_bin']['e_min'])
        result = 0.5 * np.log(e_max*e_min)
        return result

    def _log_ref_energy(self):
        return np.log(self.config['grid']['energy_bin']['e_ref'])

    def _log_rel_ebin_centers(self):
        res = self._log_ebin_centers() - self._log_ref_energy()
        return res

    def _log_dE(self):
        log_e = self._log_rel_ebin_centers()
        return log_e[1:]-log_e[:-1]
