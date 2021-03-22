import nifty7 as ift
import numpy as np

def makePSFmodel(psf_field):
    position_space = psf_field.domain 
    priors_psf = {'offset_mean': 0.01,
                'offset_std': (1e-2, 1e-4),
                'fluctuations':(1, 0.1),
                'loglogavgslope':(-4, 0.4),
                'flexibility':(1.0, 0.5),
                'asperity': None,
                'prefix': 'psf'
    }
    psf_model = ift.SimpleCorrelatedField(position_space, **priors_psf)
    psf_model = psf_model.exp()
    likelihood = ift.PoissonianEnergy(psf_field) @ psf_model
    return likelihood, psf_model

def minimizePSF(psf_likelihood, iterations = 50):
    ic_newton = ift.AbsDeltaEnergyController(name='PSF_Newton', deltaE=0.5, iteration_limit=iterations, convergence_level=3)
    minimizer = ift.NewtonCG(ic_newton)
    H = ift.StandardHamiltonian(psf_likelihood)
    pos = 0.1 * ift.from_random(H.domain)

    H = ift.EnergyAdapter(pos, H, want_metric=True)
    H, _ = minimizer(H)
    #TODO FIXME MGVI!!! 
    return H.position
