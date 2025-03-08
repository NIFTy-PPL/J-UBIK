import jax
import jax.numpy as jnp


def cast_to_dtype(tree, dtype=jnp.float32):
    return jax.tree.map(lambda x: x.astype(dtype), tree)
