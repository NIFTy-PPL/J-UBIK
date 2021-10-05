import nifty8 as ift
import numpy as np
from lib.utils import makePositiveSumPrior
from nifty8.operators.simple_linear_operators import _SlowFieldAdapter

def makePSFmodel(psf_field):
    position_space = psf_field.domain 
    priors_psf = {'offset_mean': 0,
                'offset_std': (1, 0.5),
                'fluctuations':(5.0, 2.5),
                'loglogavgslope':(-0.9, 1.0),
                'flexibility':(1, 0.3),
                'asperity': None,
                'prefix': 'psf'
    }
    center = ift.InverseGammaOperator(position_space, alpha =1.0, q=1e-4).ducktape('center')
    psf_model = ift.SimpleCorrelatedField(position_space, **priors_psf)
    psf_model = psf_model.exp() + center
    likelihood = ift.PoissonianEnergy(psf_field) @ psf_model
    return likelihood, psf_model

def minimizePSF(psf_likelihood, iterations = 50):
    ic_newton = ift.AbsDeltaEnergyController(name='PSF_Newton', deltaE=0.5, iteration_limit=iterations, convergence_level=3)
    ic_sampling = ift.AbsDeltaEnergyController(name='Samplig(lin)',deltaE=0.05, iteration_limit = 100)
    minimizer = ift.NewtonCG(ic_newton)
    H = ift.StandardHamiltonian(psf_likelihood, ic_sampling)
    pos = 0.1 * ift.from_random(H.domain)
    minimizer_sampling = ift.NewtonCG(ift.AbsDeltaEnergyController(name="Sampling (nonlin)",
                                                               deltaE=0.5, convergence_level=2,
                                                               iteration_limit= 10))
    if True:
        KL = ift.GeoMetricKl(pos, H, 2, minimizer_sampling, True)
        KL, _ = minimizer(KL)
        samples = list(KL.samples)
        return samples
    else:
        H = ift.EnergyAdapter(pos, H, want_metric=True)
        H, _ = minimizer(H)
        return H.position

def makeModularModel(psf_trainset, n_modes = 10):
    psf_samples = psf_trainset['psf_sim']
    source_samples = psf_trainset['source']

    s_dom = psf_samples[0].domain[0]
    n_of_s = len(psf_samples)
    n_dom = ift.UnstructuredDomain(n_of_s)
    in_dom = ift.makeDomain((n_dom, s_dom))

    psf_arr = []
    source_arr = []
    for i in range(n_of_s):
        psf_arr.append(psf_samples[i].val)
        source_arr.append(source_samples[i].val)
    psf_arr = np.array(psf_arr)
    source_arr = np.array(source_arr)
    psf_arr.shape = source_arr.shape = in_dom.shape

    psf_field = ift.Field.from_raw(in_dom, psf_arr)
    source_field = ift.Field.from_raw(in_dom, source_arr)


    FFT = ift.FFTOperator(in_dom, space=1)
    FFT_small = ift.FFTOperator(s_dom)
    spread = ift.ContractionOperator(in_dom, 0).adjoint
    h_in_dom = FFT.target
    spread_har = ift.ContractionOperator(h_in_dom, 0).adjoint

    signal = ift.FieldAdapter(in_dom, 'signals').real
    #amp_p = makePositiveSumPrior(s_dom, n_modes)
    sum = None
    for i in range(n_modes):
       # amp = spread @ _SlowFieldAdapter(amp_p.target, f'amp_{i}').real @ amp_p
        amp = spread @ ift.exp(ift.FieldAdapter(s_dom, f'amp_{i}')).real
        p = spread_har @ FFT_small @ ift.exp(ift.FieldAdapter(s_dom, f'psf_{i}')).real
        pointwise = signal * amp
        #FIXME use less FFT's
        #FIXME break PBC's
        prod = p * FFT(pointwise)
        if sum == None: # make this nice and shiny
            sum = prod
        else:
            sum = sum + prod
    
    modular_model = FFT.inverse(sum).real

    likelihood = ift.PoissonianEnergy(psf_field) @ modular_model
    ic_newton = ift.AbsDeltaEnergyController(name='PSF_Newton', deltaE=0.5, iteration_limit=20, convergence_level=3)
    minimizer = ift.NewtonCG(ic_newton)
    H = ift.StandardHamiltonian(likelihood)

    pos = 0.1 * ift.from_random(H.domain)
    dct = ift.MultiField.to_dict(pos)
    dct.pop('signals')
    pos = ift.MultiField.from_dict(dct)

    signal_mf = ift.MultiField.from_dict({'signals': source_field})
    pos = ift.MultiField.union((pos, signal_mf))

    pl = ift.Plot()
    pl.add(psf_field.val)

    H = ift.EnergyAdapter(pos, H, want_metric=True, constants='signals')
    H, _ = minimizer(H)
    pos = ift.MultiField.union((H.position, signal_mf))
    #TODO FIXME MGVI!!! 
    return pos, modular_model, psf_field
