import os
import dataclasses
from typing import Any

import nifty8.re as jft

from .data import  create_data_from_config, load_masked_data_from_config
from .response import build_erosita_response_from_config
from .utils import get_config



def generate_erosita_likelihood_from_config(config_file_path, prepend_ops):
    """ Creates the eROSITA Poissonian log-likelihood given the path to the config file.

    Parameters
    ----------
    config_file_path : string
        Path to config file
    Returns
    -------
    poissonian: jft.Likelihood
        Poissoninan likelihood for the eROSITA data and response, specified in the config
    """

    # load config
    response_dict = build_erosita_response_from_config(config_file_path)

    # Create data files
    create_data_from_config(config_file_path, response_dict)
    # Load data files
    masked_data = load_masked_data_from_config(config_file_path)
    response_func = response_dict['R']


    class FullModel(jft.Model):
        kern: Any = dataclasses.field(metadata=dict(static=False))
        def __init__(self, kern, instrument, pre_ops):
            self.instrument = instrument
            self.kern = kern
            self.pre_ops = pre_ops
            super().__init__(init=self.pre_ops.init)

        def __call__(self, x):
            return self.instrument(x=self.pre_ops(x), k=self.kern)

    full_model = FullModel(kern=response_dict["kernel_arr"],
                           instrument=response_func,
                           pre_ops=prepend_ops)
    return jft.Poissonian(masked_data).amend(full_model)
