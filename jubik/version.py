# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2026 Max-Planck-Society

# %%

from importlib.metadata import version

# Single source: `version` in pyproject.toml. Kept static there (not read from
# this file) so installers see the metadata without building jubik, which
# downstream `uv` needs for `match-runtime` on the jax/jaxlib build requirement.
__version__ = version("jubik")
