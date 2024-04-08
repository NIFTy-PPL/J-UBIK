# Authors: Vincent Eberle, Margret Westerkamp, Philipp Frank

from functools import partial, reduce
import jax
import jax.numpy as jnp

import nifty8.re as jft


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
        return (jax.vmap(self._model, in_axes=self._axs,
                         out_axes=self._out_axs)(x)).reshape(self._shape)


class GeneralModel(jft.Model):
    """#FIXME Some Docstring."""

    def __init__(self, dict_of_fields={}):
        """Initziales the general sky model.
        #NOTE Write some text, which of the keys are allowed.

        Paramters:
        ----------
        dict of fields: the respective keys and the
            nifty.re.models as values"""
        self._available_fields = dict_of_fields

    def build_model(self):
        """#NOTE Docstring."""
        def add_functions(f1, f2):
            def function(x):
                return f1(x) + f2(x)
            return function

        if 'spatial' not in self._available_fields.keys() or self._available_fields['spatial'] is None:
            raise NotImplementedError
        else:
            spatial = self._available_fields['spatial']
            func = spatial
            domain = spatial.domain
            if 'freq_plaw' in self._available_fields.keys() and self._available_fields['freq_plaw'] is not None:
                plaw = self._available_fields['freq_plaw']
                func = add_functions(func, plaw)
                domain = domain | plaw.domain
            if 'freq_dev' in self._available_fields.keys() and self._available_fields['freq_dev'] is not None:
                dev = self._available_fields['freq_dev']

                def extract_keys(a, domain):
                    b = {key: a[key] for key in domain}
                    return b

                def extracted_dev(op):
                    def callable_dev(x):
                        return op(extract_keys(x, op.domain))
                    return callable_dev

                func = add_functions(func, extracted_dev(dev))
                domain = domain | dev.domain
            if 'pol' in self._available_fields.keys() and self._available_fields['pol'] is not None:
                raise NotImplementedError
            if 'time' in self._available_fields.keys() and self._available_fields['time'] is not None:
                raise NotImplementedError
            res = jft.Model(func, domain=domain)
        return res


def build_power_law(freqs, alph):
    """Models a power law. Building bloc for e.g. a multifrequency model"""
    if isinstance(alph, jft.Model):
        res = lambda x: jnp.outer(freqs, alph(x)).reshape(freqs.shape + alph.target.shape)
    elif isinstance(alph, float):
        # FIXME not working at the moment
        res = jnp.outer(freqs, alph).reshape(freqs.shape)
        return jft.Model(res, domain=alph.domain)
