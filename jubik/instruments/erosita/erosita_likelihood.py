# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp
#
# Copyright(C) 2024 Max-Planck-Society
#
# %%

import dataclasses
from typing import Any

import nifty.re as jft

from .erosita_data import create_erosita_data_from_config
from .erosita_response import build_erosita_response_from_config
from ...data import load_masked_data_from_config


def generate_erosita_likelihood_from_config(config,
                                            prepend_operator):
    """ Creates the eROSITA Poissonian log-likelihood given the path to the
    config file.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration parameters.
    prepend_operator : Union[Callable, jft.Model]
        Operator to be prepended to the likelihood chain.

    Returns
    -------
    poissonian: jft.Likelihood
        Poissoninan likelihood for the eROSITA data and response, specified
        in the config.
    """

    # load config
    response_dict = build_erosita_response_from_config(config)

    # Create data files
    create_erosita_data_from_config(config, response_dict)

    # Load data files
    masked_data = load_masked_data_from_config(config)

    return generate_erosita_likelihood(response_dict,
                                       masked_data,
                                       prepend_operator)


def generate_erosita_likelihood(response_dict,
                                masked_data,
                                prepend_operator):
    """ Creates the eROSITA Poissonian log-likelihood given the path to the
    config file.

    Parameters
    ----------
    response_dict : dict
        Dictionary containing the response function and the kernel
    masked_data : jft.Vector
        Vector of masked eROSITA (mock) data for each TM
    prepend_operator : Union[Callable, jft.Model]
        Operator to be prepended to the likelihood chain.

    Returns
    -------
    poissonian: jft.Likelihood
        Poissoninan likelihood for the eROSITA data and response, specified
        in the config.
    """

    class FullModel(jft.Model):
        kern: Any = dataclasses.field(metadata=dict(static=False))

        def __init__(self, kern, instrument, pre_ops):
            self.instrument = instrument
            self.kern = kern
            self.pre_ops = pre_ops
            super().__init__(init=self.pre_ops.init)

        def __call__(self, x):
            return self.instrument(x=self.pre_ops(x), k=self.kern)

    full_model = FullModel(kern=response_dict["kernel"],
                           instrument=response_dict["R"],
                           pre_ops=prepend_operator)

    return jft.Poissonian(masked_data).amend(full_model)
