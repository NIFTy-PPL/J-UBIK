import nifty8 as ift
import xubik0 as xu
from matplotlib.colors import LogNorm

import nifty8.re as jft
import xubik0 as xu
from jax import numpy as jnp


def create_sky_model_from_config(config_file_path):
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
    if not isinstance(config_file_path, str):
        raise TypeError("The path to the config file needs to be a string")
    if not config_file_path.endswith('.yaml'):
        raise ValueError("The sky model parameters need to be safed in a .yaml-file.")

    config = xu.get_config(config_file_path)
    priors = config['priors']
    grid_info = config['grid']
    tel_info = config['telescope']

    return create_sky_model(grid_info['npix'], grid_info['padding_ratio'],
                            tel_info['fov'], priors)


def create_sky_model(npix, padding_ratio, fov, priors):
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
    space_shape = 2 * (npix, )
    distances = fov / npix

    diffuse_component, pspec = create_diffuse_component_model(space_shape,
                                                              padding_ratio,
                                                              distances,
                                                              priors['diffuse']['offset'],
                                                              priors['diffuse']['fluctuations'],
                                                              priors['diffuse']['prefix'])
    if priors['point_sources'] is None:
        sky = diffuse_component
        sky_dict = {'sky': sky, 'pspec': pspec}
    else:
        point_sources = create_point_source_model(space_shape, **priors['point_sources'])
        sky = fuse_model_components(diffuse_component, point_sources)
        sky_dict = {'sky': sky, 'point_sources': point_sources, 'diffuse': diffuse_component,
                    'pspec': pspec}
    return sky_dict


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


def create_diffuse_component_model(shape, padding_ratio, distances, offset, fluctuations, prefix='diffuse_'):
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
    ext_shp = tuple(int(entry * padding_ratio) for entry in shape)
    cfm = jft.CorrelatedFieldMaker(prefix=prefix)
    cfm.set_amplitude_total_offset(**offset)
    cfm.add_fluctuations(ext_shp, distances, **fluctuations, non_parametric_kind='power')
    cf = cfm.finalize()
    pspec = cfm.power_spectrum

    exp_padding = lambda x: jnp.exp(cf(x)[:shape[0],:shape[1]])
    diffuse = jft.Model(exp_padding, domain=cf.domain)
    return diffuse, pspec


def create_point_source_model(shape, alpha, q, key='points'):
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
    point_sources = jft.invgamma_prior(a=alpha, scale=q)
    points_func = lambda x: point_sources(x[key])
    return jft.Model(points_func, domain={key: jft.ShapeWithDtype(shape)})


# FIXME: DELETE SKY MODEL CLASS

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
        if not isinstance(config_file, str):
            raise TypeError("The config_file argument needs to be the path to a .yaml config file.")
        # FIXME: add all relevant checks and docstrings

        # Load config
        self.config = xu.get_config(config_file)
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
        """
        returns a dictionary containing:
            - sky-operator
            - power-spectrum-operator
        if point-sources are switched on:
            - point-sources
            - diffuse (correlated field)
        """
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
