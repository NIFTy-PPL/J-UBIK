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

    def __init__(self, config_file):

        """Gets the parameters needed for building the sky model from the config file
        given the corresponding path and uses these to create the sky model.

        Parameters
        ----------
        config_file_path : string
            Path to the config file

        Returns
        -------
        sky_dict: dict
            Dictionary of sky component models
        """
        if not isinstance(config_file, str):
            raise TypeError("The path to the config file needs to be a string")
        if not config_file.endswith('.yaml'):
            raise ValueError("The sky model parameters need to be safed in a .yaml-file.")

        self.config = ju.get_config(config_file)
        self.diffuse_component = None
        self.point_sources = None
        self.pspec = None
        self.sky = None

    def create_sky_model(self, sdim=None, edim=None, padding_ratio=None,
                         fov=None, energy_range= None, priors=None):
        """Returns a dictionary of sky component models given the grid information
        (# pixels and padding ration) and the telescope information (FOV: field of view) as well as a
        dictionary for the prior parameters

        Parameters
        ----------
        npix: int
            Number of pixels in each direction
        padding_ratio: float
            Ratio between number of pixels in the actual space and the padded space
        fov: float
            FOV of the telescope
        priors: dict
            Dictionary of prior parameters for the correlated field needed and for the point sources optional
            in the format:
                    priors:
                        point_sources:
                            alpha: optional
                            q: optional
                        diffuse:
                            offset:
                                offset_mean:
                                offset_std:
                            fluctuations:
                                fluctuations:
                                loglogavgslope:
                                flexibility:
                                asperity:
                            prefix:

        Returns
        -------
        sky_dict: dict
            Dictionary of sky component models
        """
        if sdim is None:
            sdim = self.config['grid']['sdim']
        if edim is None:
            edim = self.config['grid']['edim']
        if padding_ratio is None:
            npix = self.config['grid']['padding_ratio']
        if fov is None:
            fov = self.config['telescope']['fov']
        if energy_range is None:
            e_min = self.config['grid']['energy_bin']['e_min']
            e_max = self.config['grid']['energy_bin']['e_max']
            energy_range = e_max - e_min
        if priors is None:
            priors = self.config['priors']

        sdim = 2 * (sdim,)
        sdistances = fov / sdim
        edistances = energy_range/ edim

        self.diffuse_component, self.pspec = self._create_diffuse_component_model(sdim, edim,
                                                                  padding_ratio,
                                                                  edistances,
                                                                  priors['diffuse'])
        if 'point_sources' not in priors:
            self.sky = self.diffuse_component
        else:
            self.point_sources = self._create_point_source_model(sdim,
                                                                 **priors['point_sources'])
            self.sky = fuse_model_components(self.diffuse_component, self.point_sources)

    def _create_correlated_field(self, shape, distances, prior_dict):
        cfm = jft.CorrelatedFieldMaker(prefix=prior_dict['space']['prefix'])
        cfm.set_amplitude_total_offset(**prior_dict['space']['offset'])
        cfm.add_fluctuations(shape, distances, **prior_dict['space']['fluctuations'],
                             non_parametric_kind='power')
        cf = cfm.finalize()
        return cf, cfm.power_spectrum()

    def _create_diffuse_component_model(self, sdim, edim, padding_ratio, sdistances,
                                        edistances, prior_dict):
        """ Returns a model for the diffuse component given the information on its shape and
        distances and the prior dictionaries for the offset and the fluctuations

        Parameters
        ----------
        shape : tuple of int or int
            Position-space shape.
        distances : tuple of float or float
            Position-space distances
        offset: dict
            Prior dictionary for the offset of the diffuse component of the form
                offset_mean: float
                offset_std: (float, float)
        fluctuations: dict
            Prior dictionary for the fluctuations of the diffuse component of the form
                fluctuations: (float, float)
                loglogavgslope: (float, float)
                flexibility: (float, float)
                asperity: (float, float)
        prefix: string
            Prefix for the power spectrum parameter domain names

        Returns
        -------
        diffuse: jft.Model
            Model for the diffuse component
        pspec: Callable
            Power spectrum
        """
        if not 'space' in prior_dict:
            return ValueError('Ever diffuse component needs a spatial component')
        ext_shp = tuple(int(entry * padding_ratio) for entry in sdim)
        self.spatial_cf, self.spatial_pspec = self._create_correlated_field(ext_shp,
                                                                            sdistances,
                                                                            prior_dict['space'])
        if 'plaw' in prior_dict:
            alpha_cf, _ = self._create_correlated_field(ext_shp, sdistances, prior_dict['plaw'])
            self.plaw = ju.build_power_law(jnp.arange(0, edim, 1), alpha_cf)
        if 'dev' in prior_dict:
            self.dev_cf, self.dev_pspec = self._create_correlated_field(edim,
                                                                        edistances,
                                                                        prior_dict['dev'])

        log_diffuse = ju.GeneralModel({'spatial': self.spatial_cf,
                                       'freq_plaw': self.plaw,
                                       'freq_dev': self.dev}).build_model()
        exp_padding = lambda x: jnp.exp(log_diffuse(x)[:sdim[0], :sdim[1]])
        self.diffuse = jft.Model(exp_padding, domain=log_diffuse.domain)
        return self.diffuse

    def _create_point_source_model(self, sdim, edim, sdistances, edistances, prior_dict):
        """ Returns a model for the point-source component given the information on its shape
         and information on the shape and scaling parameters

        Parameters
        ----------
        shape : tuple of int or int
            Position-space shape.
        alpha: float
            Inverse gamma shape parameter
        q: float
            Inverse gamma scaling parameter
        key: string
            Prefix for the point-source parameter domain names

        Returns
        -------
        points: jft.Model
            Model for the point-source component
        """
        if not 'space' in prior_dict:
            return ValueError('Point source component needs a spatial component')
        point_sources = jft.invgamma_prior(a=prior_dict['spatial']['alpha'],
                                           scale=prior_dict['spatial']['q'])
        points_func = lambda x: point_sources(x[prior_dict['spatial']['key']])
        self.points_invg = jft.Model(points_func,
                                    domain={prior_dict['spatial']['key']: jft.ShapeWithDtype(sdim)})
        if 'plaw' in prior_dict:
            points_alpha_cf, _ = self._create_correlated_field(sdim, sdistances, prior_dict['plaw'])
            self.plaw = ju.build_power_law(jnp.arange(0, edim, 1), points_alpha_cf)
        if 'dev' in prior_dict:
            self.points_dev_cf, self.points_dev_pspec = self._create_correlated_field(edim,
                                                                                      edistances,
                                                                                      prior_dict['dev'])
