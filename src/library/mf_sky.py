# Authors: Vincent Eberle, Margret Westerkamp, Philipp Frank

from functools import partial
import jax
import nifty8.re as jft


class MappedModel(jft.Model):
    """Mapping of a CF Model.

    This class builds a forward model "ndof" copies of a
    correlated field model. The correlated fields share the
    same power spectrum but will have different
    "xi"s.
    """

    def __init__(self, correlated_field, cf_prefix, ndof, first_axis=True):
        """Initialise the mapping.

        Paramters:
        ----------
        correlated_field : CF Model
        cf_prefix: probably string
        ndof: int, number of copies
        first_axis: if True prepends the ndof copies els appends
        """
        self._cf = correlated_field
        keys = correlated_field.domain.keys()
        xi_key = cf_prefix+'xi'
        if xi_key not in keys:
            raise ValueError

        xi_dom = correlated_field.domain[xi_key]
        if first_axis:
            new_primals = jft.ShapeWithDtype((ndof,) + xi_dom.shape, xi_dom.dtype)
            axs = 0
            self._out_axs = 0
        else:
            new_primals = jft.ShapeWithDtype(xi_dom.shape + (ndof,), xi_dom.dtype)
            axs = -1
            self._out_axs = 1

        new_domain = correlated_field.domain.copy()
        new_domain[xi_key] = new_primals

        xiinit = partial(jft.random_like, primals=new_primals)

        init = correlated_field.init
        init = {k: init[k] if k != xi_key else xiinit for k in keys}

        self._axs = ({k: axs if k == xi_key else None for k in keys},)
        super().__init__(domain=new_domain, init=jft.Initializer(init))

    def __call__(self, x):
        x = x.tree if isinstance(x, jft.Vector) else x
        return jax.vmap(self._cf,
                        in_axes=self._axs,
                        out_axes=self._out_axs
                        )(x)


def mf_model(freqs, alph, spatial, dev):
    """Typical Multifrequency Model for physical forward modelling.

    This models physical field which is correlated spatially and its
    Energy follows a power law function with spectral indices alpha,
    which can be a field (correlated or not, spatially) with and offset
    (value of the spatial field) and deviations on the powerlaw (dev)

    # TODO for iregular gridded freqs, the correlated field model cannot be
    used for the deviations

    """
    if isinstance(alph, jft.Model):
        plaw = lambda x: jnp.outer(freqs, alph(x)).reshape(freqs.shape + alph.target.shape)
        plaw_offset = lambda x: plaw(x) + spatial(x)
        res = lambda x: plaw_offset(x) + dev(x)
        domain = alph.domain | spatial.domain | dev.domain
        res = jft.Model(res, domain=domain)
    elif isinstance(alph, float):
        res = jnp.outer(freqs, alph).reshape(freqs.shape)
    return res

# import nifty8 as ift
# from jax import random
# import jax.numpy as jnp
# import numpy as np
# import jax
# from nifty8.re.tree_math import ShapeWithDtype

# # Random Stuff
# seed = 42
# key = random.PRNGKey(seed)
# key, subkey = random.split(key)

# dims = (10)

# # Correlated Field Energy 1D
# cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
# cf_fl = {
#     "fluctuations": (1e-1, 5e-3),
#     "loglogavgslope": (-1., 1e-2),
#     "flexibility": (1e+0, 5e-1),
#     "asperity": (5e-1, 5e-2),
#     "harmonic_type": "Fourier"
# }
# cfm = jft.CorrelatedFieldMaker("cf")
# cfm.set_amplitude_total_offset(**cf_zm)
# cfm.add_fluctuations(
#     dims,
#     distances=1. / dims,
#     **cf_fl,
#     prefix="ax1",
#     non_parametric_kind="power"
# )
# correlated_field = cfm.finalize()

# # Initial Position for 1D Field
# pos_init = jft.Vector(jft.random_like(subkey, cfm._parameter_tree))
# print(correlated_field(pos_init).shape)

# # Partial dictionary Evaluation for enabling vmap
# def partial_apply_cf(cf, pos_init):
#     def partly(xi, xi_key):
#         pos_init.tree.pop(xi_key)
#         pos_init.tree.update({xi_key: xi})
#         return cf(pos_init)
#     return partly


# func = partial_apply_cf(correlated_field, pos_init)

# # Bigger parametertree
# dict = cfm._parameter_tree # = domain for CF
# dict.update({"cfxi": ShapeWithDtype(shape=(10, 200))})

# # New position, but bigger
# new_pos_init = jft.Vector(jft.random_like(subkey, dict))

# # The magic line
# cf_list = jax.vmap(func, in_axes=(1, None), out_axes=1)#(new_pos_init.tree['cfxi'], 'cfxi')
# print(cf_list)
# res = cf_list(new_pos_init.tree['cfxi'], 'cfxi')
# jft.Model(cf_list, domain=dict)
# print(res.shape)
