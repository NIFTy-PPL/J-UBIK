import operator
from functools import partial, reduce
import jax
import jax.numpy as jnp

class StokesAdder(jft.Model):
    def __init__(self, correlated_field_dict):
        '''
            Takes in a dict for pre-stokes fields a,b,c,d and outputs the stokes fields I,Q,U,V.

            The relation is I = exp(a)*cosh(p), Q,U,V = exp(a)*sinh(p)* b/p,c/p,d/p with p = sqrt(b^2 + c^2 + d^2).
        '''
        self.cfs = correlated_field_dict
        
        super().__init__(init=reduce(operator.or_,
                                     [value.init for value in self.cfs.values()]))

    def __call__(self, x):
        def get_stokes(pre_stokes):
            pol_int = jnp.sqrt(sum(pre_stokes[i]**2 for i in range(1,4)))
            return jnp.concatenate([jnp.exp(pre_stokes[:1])*jnp.cosh(pol_int),
                                    (jnp.exp(pre_stokes[:1])*jnp.sinh(pol_int)/pol_int)*pre_stokes[1:]])

        pre_stokes = jnp.stack([cf(x) for cf in self.cfs.values()])
        dims_remaining = pre_stokes.shape[1:]
        pre_stokes = pre_stokes.reshape((4,-1))

        stokes = jax.vmap(get_stokes, in_axes=1, out_axes=-1)(pre_stokes)
        return stokes.reshape((4,) + dims_remaining)