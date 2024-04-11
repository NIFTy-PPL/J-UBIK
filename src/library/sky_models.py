import nifty8 as ift
from matplotlib.colors import LogNorm

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

    def __init__(self, config_file_path):

        """Gets the parameters needed for building the sky model from the config file
        given the corresponding path.

        Parameters
        ----------
        config_file_path : string
            Path to the config file
        """
        if not isinstance(config_file_path, str):
            raise TypeError("The path to the config file needs to be a string")
        if not config_file_path.endswith('.yaml'):
            raise ValueError("The sky model parameters need to be safed in a .yaml-file.")

        self.config = ju.get_config(config_file_path)
        self.diffuse = None
        self.point_sources = None
        self.pspec = None
        self.sky = None

        self.plaw = None
        self.dev_cf = None
        self.dev_pspec = None

        self.points_plaw = None
        self.points_dev_cf = None
        self.points_dev_pspec = None

    def create_sky_model(self, sdim=None, edim=None, s_padding_ratio=None,
                         e_padding_ratio=None,
                         fov=None, energy_range= None, priors=None):
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
        if sdim is None:
            sdim = self.config['grid']['sdim']
        if edim is None:
            edim = self.config['grid']['edim']
        if s_padding_ratio is None:
            s_padding_ratio = self.config['grid']['s_padding_ratio']
        if e_padding_ratio is None:
            e_padding_ratio = self.config['grid']['e_padding_ratio']
        if fov is None:
            fov = self.config['telescope']['fov']
        if energy_range is None:
            e_min = self.config['grid']['energy_bin']['e_min']
            e_max = self.config['grid']['energy_bin']['e_max']
            energy_range = e_max - e_min
        if priors is None:
            priors = self.config['priors']

        sdim = 2 * (sdim,)
        sdistances = fov / sdim[0]
        edistances = energy_range/ edim

        _ = self._create_diffuse_component_model(sdim, edim, s_padding_ratio, e_padding_ratio,
                                                 sdistances, edistances, priors['diffuse'])
        if 'point_sources' not in priors:
            self.sky = self.diffuse
        else:
            _ = self._create_point_source_model(sdim, edim, e_padding_ratio,
                                                sdistances, edistances, priors['point_sources'])
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

    def _create_diffuse_component_model(self, sdim, edim, s_padding_ratio, e_padding_ratio,
                                        sdistances,
                                        edistances, prior_dict):
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
        ext_s_shp = tuple(int(entry * s_padding_ratio) for entry in sdim)
        ext_e_shp = int(edim * e_padding_ratio)
        self.spatial_cf, self.spatial_pspec = self._create_correlated_field(ext_s_shp,
                                                                            sdistances,
                                                                            prior_dict['spatial'])
        if 'plaw' in prior_dict:
            alpha_cf, _ = self._create_correlated_field(ext_s_shp, sdistances, prior_dict['plaw'])
            self.plaw = ju.build_power_law(jnp.arange(0, ext_e_shp, 1), alpha_cf)
        if 'dev' in prior_dict:
            dev_cf, self.dev_pspec = self._create_correlated_field(ext_e_shp,
                                                                   edistances,
                                                                   prior_dict['dev'])
            self.dev_cf = ju.MappedModel(dev_cf, prior_dict['dev']['prefix']+'xi',
                                         ext_s_shp, False)

        log_diffuse = ju.GeneralModel({'spatial': self.spatial_cf,
                                       'freq_plaw': self.plaw,
                                       'freq_dev': self.dev_cf}).build_model()
        exp_padding = lambda x: jnp.exp(log_diffuse(x)[:edim, :sdim[0], :sdim[1]])
        self.diffuse = jft.Model(exp_padding, domain=log_diffuse.domain)
        return self.diffuse

    def _create_point_source_model(self, sdim, edim, e_padding_ratio,
                                   sdistances, edistances, prior_dict):
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
        sdistances : tuple of float or float
            Position-space distances
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
        if not 'spatial' in prior_dict:
            return ValueError('Point source component needs a spatial component')
        ext_e_shp = int(edim * e_padding_ratio)
        point_sources = jft.invgamma_prior(a=prior_dict['spatial']['alpha'],
                                           scale=prior_dict['spatial']['q'])
        points_func = lambda x: point_sources(x[prior_dict['spatial']['key']])
        self.points_invg = jft.Model(points_func,
                                    domain={prior_dict['spatial']['key']: jft.ShapeWithDtype(sdim)})
        if 'plaw' in prior_dict:
            points_alpha_cf, _ = self._create_correlated_field(sdim, sdistances, prior_dict['plaw'])
            self.points_plaw = ju.build_power_law(jnp.arange(0, ext_e_shp, 1), points_alpha_cf)
        if 'dev' in prior_dict:
            points_dev_cf, self.points_dev_pspec = self._create_correlated_field(ext_e_shp,
                                                                                 edistances,
                                                                                 prior_dict['dev'])
            self.points_dev_cf = ju.MappedModel(points_dev_cf, prior_dict['dev']['prefix']+'xi',
                                                sdim, False)

        log_points = ju.GeneralModel({'spatial': self.points_invg,
                                       'freq_plaw': self.points_plaw,
                                       'freq_dev': self.points_dev_cf}).build_model()
        exp_padding = lambda x: jnp.exp(log_points(x)[:edim, :, :])
        self.point_sources = jft.Model(exp_padding, domain=log_points.domain)
        return self.point_sources

