# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp, Philipp Frank

# Copyright(C) 2024 Max-Planck-Society

# %%
from functools import partial, reduce

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from ducc0.fft import good_size as good_fft_size

from .utils import add_functions, add_models


class SkyModel:
    """
    Basic spatial SkyModel.

    This SkyModel consists of a diffuse (correlated) component
    and a point-source like (not correlated) component. The latter can
    be switched off.
    """

    def __init__(self, config: dict = None):
        """Initializes the SkyModel with the provided config dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration parameters.
        """
        if config is not None:
            if not isinstance(config, dict):
                raise TypeError("The config must be a python-dictionary.")
            self.config = config
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

    def create_sky_model(
        self,
        sdim=None,
        edim=None,
        s_padding_ratio=None,
        e_padding_ratio=None,
        fov=None,
        e_min=None,
        e_max=None,
        e_ref=None,
        priors=None,
    ):
        """Returns the sky model composed out of components.

        This returns the sky model given the grid information
        (# pixels and padding ration), the telescope information (FOV: field of
        view) as well as a dictionary for the prior parameters. All these
        parameters can be set externally or take the SkyModels config file,
        if they are set to None.

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
        if "grid" not in self.config.keys():
            self.config["grid"] = {}
        if "telescope" not in self.config.keys():
            self.config["telescope"] = {}
        if sdim is None:
            sdim = self.config["grid"]["sdim"]
        else:
            self.config["grid"]["sdim"] = sdim
        if edim is None:
            edim = self.config["grid"]["edim"]
        else:
            self.config["grid"]["edim"] = edim
        if s_padding_ratio is None:
            s_padding_ratio = self.config["grid"]["s_padding_ratio"]
        else:
            self.config["grid"]["s_padding_ratio"] = s_padding_ratio
        if e_padding_ratio is None:
            e_padding_ratio = self.config["grid"]["e_padding_ratio"]
        else:
            self.config["grid"]["e_padding_ratio"] = e_padding_ratio
        if fov is None:
            fov = self.config["telescope"]["fov"]
        else:
            self.config["telescope"]["fov"] = fov
        if "energy_bin" not in self.config["grid"].keys():
            self.config["grid"]["energy_bin"] = {}
        if e_min is None:
            e_min = self.config["grid"]["energy_bin"]["e_min"]
        else:
            self.config["grid"]["energy_bin"]["e_min"] = e_min
        if e_max is None:
            e_max = self.config["grid"]["energy_bin"]["e_max"]
        else:
            self.config["grid"]["energy_bin"]["e_max"] = e_max
        if e_ref is not None:
            self.config["grid"]["energy_bin"]["e_ref"] = e_ref
        if priors is None:
            priors = self.config["priors"]
        else:
            self.config["priors"] = priors

        sdim = 2 * (sdim,)
        self.s_distances = fov / sdim[0]
        energy_range = np.array(e_max) - np.array(e_min)
        self.e_distances = (
            energy_range / edim
        )  # FIXME: add proper distances for irregular energy grid

        if (
            not isinstance(self.e_distances, float)
            and "dev_corr" in priors["diffuse"].keys()
        ):
            raise ValueError(
                "Grid distances in energy direction have to be regular and defined by"
                "one float of a corrlated field in energy direction is taken."
            )

        self._create_diffuse_component_model(
            sdim,
            edim,
            s_padding_ratio,
            e_padding_ratio,
            self.s_distances,
            self.e_distances,
            priors["diffuse"],
        )
        if "point_sources" not in priors:
            self.sky = self.diffuse
        else:
            if "dev_corr" in priors["point_sources"].keys():
                raise ValueError(
                    "Grid distances in energy direction have to be regular and defined by"
                    "one float of a corrlated field in energy direction is taken."
                )
            self._create_point_source_model(
                sdim, edim, e_padding_ratio, self.e_distances, priors["point_sources"]
            )
            self.sky = add_models(self.diffuse, self.point_sources)
        return self.sky

    def _create_correlated_field(self, shape, distances, prior_dict):
        """Returns a 1- or 2-dim correlated field and the corresponding
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

        cfm = jft.CorrelatedFieldMaker(prefix=prior_dict["prefix"])
        cfm.set_amplitude_total_offset(**prior_dict["offset"])
        cfm.add_fluctuations(shape, distances, **prior_dict["fluctuations"])
        cf = cfm.finalize()
        return cf, cfm.power_spectrum

    def _create_wiener_process(self, x0, sigma, dE, name, edims):
        return jft.WienerProcess(x0, tuple(sigma), dt=dE, name=name, N_steps=edims - 1)

    def _create_diffuse_component_model(
        self,
        sdim,
        edim,
        s_padding_ratio,
        e_padding_ratio,
        sdistances,
        edistances,
        prior_dict,
    ):
        """Returns a model for the diffuse component given the information on its shape and
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
        if not "spatial" in prior_dict:
            return ValueError("Every diffuse component needs a spatial component")
        if "dev_wp" in prior_dict and "dev_corr" in prior_dict:
            raise ValueError(
                "You can only inlude Wiener process or correlated field"
                "for the deviations around the plaw."
            )
        ext_s_shp = tuple(good_fft_size(int(entry * s_padding_ratio)) for entry in sdim)
        ext_e_shp = good_fft_size(int(edim * e_padding_ratio))
        self.spatial_cf, self.spatial_pspec = self._create_correlated_field(
            ext_s_shp, sdistances, prior_dict["spatial"]
        )
        if "plaw" in prior_dict:
            self.alpha_cf, self.alpa_pspec = self._create_correlated_field(
                ext_s_shp, sdistances, prior_dict["plaw"]
            )
            self.plaw = _apply_slope(self._log_rel_ebin_centers(), self.alpha_cf)

        if "dev_corr" in prior_dict:
            dev_cf, self.dev_pspec = self._create_correlated_field(
                ext_e_shp, edistances, prior_dict["dev_cor"]
            )
            self.dev_cf = MappedModel(
                dev_cf, prior_dict["dev_corr"]["prefix"] + "xi", ext_s_shp, False
            )
        if "dev_wp" in prior_dict:
            dev_cf = self._create_wiener_process(
                edims=len(self._log_rel_ebin_centers()),
                dE=self._log_dE(),
                **prior_dict["dev_wp"]
            )
            self.dev_cf = MappedModel(
                dev_cf, prior_dict["dev_wp"]["name"], ext_s_shp, False
            )
        log_diffuse = GeneralModel(
            {
                "spatial": self.spatial_cf,
                "freq_plaw": self.plaw,
                "freq_dev": self.dev_cf,
            }
        ).build_model()
        exp_padding = lambda x: jnp.exp(log_diffuse(x)[:edim, : sdim[0], : sdim[1]])
        self.diffuse = jft.Model(exp_padding, domain=log_diffuse.domain)

    def _create_point_source_model(
        self, sdim, edim, e_padding_ratio, edistances, prior_dict
    ):
        """Return a model for the point-source component.

        Given the information on its shape and information on the shape
        and scaling parameters, this returns a model for the point source
        comonent.

        Parameters
        ----------
        sdim: int or tuple of int
            Number of pixels in each spatial dimension
        edim: int
            Number of pixels in spectral direction
        e_padding_ratio: float
            Ratio between number of pixels in the actual enegery space and the
            padded energy space. It needs to be taken such that the correlated
            fields in energy direction has more than 3 pixels.
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
        if "spatial" not in prior_dict:
            return ValueError("Point source component needs a spatial component")
        if "dev_wp" in prior_dict and "dev_corr" in prior_dict:
            raise ValueError(
                "You can only inlude Wiener process or correlated field"
                "for the deviations around the plaw."
            )
        ext_e_shp = int(edim * e_padding_ratio)
        point_sources = jft.invgamma_prior(
            a=prior_dict["spatial"]["alpha"], scale=prior_dict["spatial"]["q"]
        )
        points_log_func = lambda x: jnp.log(
            point_sources(x[prior_dict["spatial"]["key"]])
        )
        self.points_log_invg = jft.Model(
            points_log_func,
            domain={prior_dict["spatial"]["key"]: jft.ShapeWithDtype(sdim)},
        )

        if "plaw" in prior_dict:
            self.points_alpha = jft.NormalPrior(
                prior_dict["plaw"]["mean"],
                prior_dict["plaw"]["std"],
                name=prior_dict["plaw"]["name"],
                shape=sdim,
                dtype=jnp.float64,
            )
            points_plaw = _apply_slope(self._log_rel_ebin_centers(), self.points_alpha)
            self.points_plaw = jft.Model(
                lambda x: points_plaw(x), domain=points_plaw.domain
            )

        if "dev_corr" in prior_dict:
            points_dev_cf, self.points_dev_pspec = self._create_correlated_field(
                ext_e_shp, edistances, prior_dict["dev_cor"]
            )
            self.points_dev_cf = MappedModel(
                points_dev_cf, prior_dict["dev_corr"]["prefix"] + "xi", sdim, False
            )
        if "dev_wp" in prior_dict:
            points_dev_cf = self._create_wiener_process(
                edims=len(self._log_rel_ebin_centers()),
                dE=self._log_dE(),
                **prior_dict["dev_wp"]
            )

            points_dev_cf = MappedModel(
                points_dev_cf, prior_dict["dev_wp"]["name"], sdim, False
            )

            self.points_dev_cf = jft.Model(
                lambda x: points_dev_cf(x), domain=points_dev_cf.domain
            )

        log_points = GeneralModel(
            {
                "spatial": self.points_log_invg,
                "freq_plaw": self.points_plaw,
                "freq_dev": self.points_dev_cf,
            }
        ).build_model()

        exp_padding = lambda x: jnp.exp(log_points(x)[:edim, :, :])
        self.point_sources = jft.Model(exp_padding, domain=log_points.domain)

    def sky_model_to_dict(self):
        """Return a dictionary with callables for the major sky models."""
        sky_dict = {
            "sky": self.sky,
            "diffuse": self.diffuse,
            "points": self.point_sources,
        }
        no_none_dict = {
            key: value for (key, value) in sky_dict.items() if value is not None
        }
        return no_none_dict

    def _log_ebin_centers(self):
        e_max = np.array(self.config["grid"]["energy_bin"]["e_max"])
        e_min = np.array(self.config["grid"]["energy_bin"]["e_min"])
        result = 0.5 * np.log(e_max * e_min)
        return result

    def _log_ref_energy(self):
        return np.log(self.config["grid"]["energy_bin"]["e_ref"])

    def _log_rel_ebin_centers(self):
        res = self._log_ebin_centers() - self._log_ref_energy()
        return res

    def _log_dE(self):
        log_e = self._log_rel_ebin_centers()
        return log_e[1:] - log_e[:-1]


class MappedModel(jft.Model):
    """Maps a model to a higher dimensional space."""

    def __init__(self, model, mapped_key, shape, first_axis=True):
        """Intitializes the mapping class.

        Parameters:
        ----------
        model: nifty.re.Model most probable a Correlated Field Model or a
            Gauss-Markov Process
        mapped_key: string, dictionary key for input dimension which is
            going to be mapped.
        shape: tuple, number of copies in each dim. Size of the
        first_axis: if True prepends the number of copies
            else they will be appended
        """
        self._model = model
        ndof = reduce(lambda x, y: x * y, shape)
        keys = model.domain.keys()
        if mapped_key not in keys:
            raise ValueError

        xi_dom = model.domain[mapped_key]
        if first_axis:
            new_primals = jft.ShapeWithDtype((ndof,) + xi_dom.shape, xi_dom.dtype)
            axs = 0
            self._out_axs = 0
            self._shape = shape + model.target.shape
        else:
            new_primals = jft.ShapeWithDtype(xi_dom.shape + (ndof,), xi_dom.dtype)
            axs = -1
            self._out_axs = 1
            self._shape = model.target.shape + shape

        new_domain = model.domain.copy()
        new_domain[mapped_key] = new_primals

        xiinit = partial(jft.random_like, primals=new_primals)

        init = model.init
        init = {k: init[k] if k != mapped_key else xiinit for k in keys}

        self._axs = ({k: axs if k == mapped_key else None for k in keys},)
        super().__init__(domain=new_domain, init=jft.Initializer(init))

    def __call__(self, x):
        x = x.tree if isinstance(x, jft.Vector) else x
        return (
            jax.vmap(self._model, in_axes=self._axs, out_axes=self._out_axs)(x)
        ).reshape(self._shape)


class GeneralModel(jft.Model):
    """General Sky Model, plugging together several components."""

    def __init__(self, dict_of_fields={}):
        """Initializes the general sky model.

        keys for the dictionary:
        -----------------------
        spatial: typically 2D Model for spatial log flux(x),
            where x is the spatial vector.
        freq_plaw: jubik0.build_power_law or other 3D model.
        freq_dev: additional flux (frequency / energy) dependent process.
            often deviations from freq_plaw.

        Parameters:
        ----------
        dict of fields: dict
            keys: str, name of the field
            val: nifty.re.Model
        """
        self._available_fields = dict_of_fields

    def build_model(self):
        """Returns Model from the dict_of_fields."""

        if (
            "spatial" not in self._available_fields.keys()
            or self._available_fields["spatial"] is None
        ):
            raise NotImplementedError
        else:
            spatial = self._available_fields["spatial"]
            func = spatial
            domain = spatial.domain
            if (
                "freq_plaw" in self._available_fields.keys()
                and self._available_fields["freq_plaw"] is not None
            ):
                plaw = self._available_fields["freq_plaw"]
                func = add_functions(func, plaw)
                domain = domain | plaw.domain
            if (
                "freq_dev" in self._available_fields.keys()
                and self._available_fields["freq_dev"] is not None
            ):
                dev = self._available_fields["freq_dev"]

                def extract_keys(a, domain):
                    b = {key: a[key] for key in domain}
                    return b

                def extracted_dev(op):
                    def callable_dev(x):
                        return op(extract_keys(x, op.domain))

                    return callable_dev

                func = add_functions(func, extracted_dev(dev))
                domain = domain | dev.domain
            res_func = (
                lambda x: func(x)
                if len(func(x).shape) == 3
                else jnp.reshape(func(x), (1,) + func(x).shape)
            )
            res = jft.Model(res_func, domain=domain)
        return res


def _apply_slope(freqs, alph):
    if isinstance(alph, jft.Model):
        res = lambda x: jnp.outer(freqs, alph(x)).reshape(
            freqs.shape + alph.target.shape
        )
    elif isinstance(alph, float):
        raise NotImplementedError
        # TODO enable inferable scalar spectral index
        # res = jnp.outer(freqs, alph).reshape(freqs.shape)
    return jft.Model(res, domain=alph.domain)
