import argparse
import os
from jax import random
import jax.numpy as jnp

import jubik0 as ju
import nifty8.re as jft

def main():

    # ############################################## CONFIG ##################################################
    # Get config filename
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    # Read dictionaries from config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    optimization_dict = cfg['optimization']
    output_directory = optimization_dict['odir']
    ju.create_output_directory(os.path.join(output_directory, 'priors'))
    utils_dict = cfg['utils']
    seed = utils_dict['seed']
    s_dims = 2*(cfg['space']['dims'],)
    e_dims = cfg['energy']['dims']

    ############################### BUILD MF model ################################################

    # Spatial Correlation
    spatial_dict = cfg['space']['priors']
    spatial_cfm = jft.CorrelatedFieldMaker(spatial_dict['fluctuations']['prefix'])
    spatial_cfm.set_amplitude_total_offset(**spatial_dict['offset'])
    spatial_cfm.add_fluctuations(
        s_dims,
        distances=1. / s_dims[0],
        **spatial_dict['fluctuations']
    )
    spatial_field = spatial_cfm.finalize()

    # Plaw
    alpha_dict = cfg['energy']['priors']['plaw']
    alpha = jft.CorrelatedFieldMaker(alpha_dict['fluctuations']['prefix'])
    alpha.set_amplitude_total_offset(**alpha_dict['offset'])
    alpha.add_fluctuations(
        s_dims,
        distances=1. / s_dims[0],
        **alpha_dict['fluctuations']
    )
    alpha_field = alpha.finalize()

    plaw = ju.build_power_law(jnp.array(cfg['energy']['freqs']), alpha_field)

    # Spectral plaw dev
    dev_dict = cfg['energy']['priors']['plaw']
    plaw_dev = jft.CorrelatedFieldMaker(dev_dict['fluctuations']['prefix'])
    plaw_dev.set_amplitude_total_offset(**dev_dict['offset'])
    plaw_dev.add_fluctuations(
        e_dims,
        distances=1. / e_dims,
        **dev_dict['fluctuations']
    )
    plaw_dev_field = alpha.finalize()

    gen_mod = ju.GeneralModel(
        {'spatial': spatial_field, 'freq_plaw': plaw, 'freq_dev': plaw_dev_field}).build_model()
    final_exp = lambda x: jnp.exp(gen_mod(x))
    final_model = jft.Model(final_exp, domain=gen_mod.domain)

    key = random.PRNGKey(seed)
    key, mockkey = random.split(key)
    pos_init = jft.Vector(jft.random_like(mockkey, final_model.domain))
    mf_prior_field = final_model(pos_init)

    key, poissonkey = random.split(key)
    mock_data = random.poisson(poissonkey, mf_prior_field)

    lklhd = jft.Poissonian(mock_data).amend(final_model)

    key, initkey = random.split(key)
    pos_init = 0.1 * jft.Vector(jft.random_like(initkey, gen_mod.domain))

    kl_solver_kwargs = optimization_dict.pop('kl_kwargs')
    kl_solver_kwargs['minimize_kwargs']['absdelta'] *= cfg['space']['dims']

    key, samplekey = random.split(key)
    samples, state = jft.optimize_kl(lklhd,
                                     pos_init,
                                     key=samplekey,
                                     kl_kwargs=kl_solver_kwargs,
                                     **optimization_dict
                                     )

if __name__ == "__main__":
    main()
