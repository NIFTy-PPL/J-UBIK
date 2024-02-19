import nifty8 as ift
import nifty8.re as jft
from jax import random
import jax.numpy as jnp
import numpy as np
import jax
from nifty8.re.tree_math import ShapeWithDtype

# Random Stuff
seed = 42
key = random.PRNGKey(seed)
key, subkey = random.split(key)

dims = (10)

# Correlated Field Energy 1D
cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
cf_fl = {
    "fluctuations": (1e-1, 5e-3),
    "loglogavgslope": (-1., 1e-2),
    "flexibility": (1e+0, 5e-1),
    "asperity": (5e-1, 5e-2),
    "harmonic_type": "Fourier"
}
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims,
    distances=1. / dims,
    **cf_fl,
    prefix="ax1",
    non_parametric_kind="power"
)
correlated_field = cfm.finalize()

# Initial Position for 1D Field
pos_init = jft.Vector(jft.random_like(subkey, cfm._parameter_tree))
print(correlated_field(pos_init).shape)

# Partial dictionary Evaluation for enabling vmap
def partial_apply_cf(cf, pos_init):
    def partly(xi, xi_key):
        pos_init.tree.pop(xi_key)
        pos_init.tree.update({xi_key: xi})
        return cf(pos_init)
    return partly


func = partial_apply_cf(correlated_field, pos_init)

# Bigger parametertree
dict = cfm._parameter_tree # = domain for CF
dict.update({"cfxi": ShapeWithDtype(shape=(10, 200))})

# New position, but bigger
new_pos_init = jft.Vector(jft.random_like(subkey, dict))

# The magic line
cf_list = jax.vmap(func, in_axes=(1, None), out_axes=1)#(new_pos_init.tree['cfxi'], 'cfxi')
print(cf_list)
res = cf_list(new_pos_init.tree['cfxi'], 'cfxi')
jft.Model(cf_list, domain=dict)
print(res.shape)
