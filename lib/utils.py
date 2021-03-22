import numpy as np
import nifty7 as ift


def get_normed_exposure_operator(exposure_field, data_array):
    norm = (data_array[exposure_field.val !=0] / exposure_field.val[exposure_field.val!=0]).mean()
    normed_exp_field = exposure_field * norm
    normed_exp_field = ift.DiagonalOperator(normed_exp_field)
    return normed_exp_field


def prior_sample_plotter(opchain, n):
    pl = ift.Plot()
    for ii in range(n):
        f = ift.from_random(opchain.domain)
        tmp = opchain(f)
        pl.add(tmp)
    return pl.output()


def get_mask_operator(exp_field):
    mask = np.zeros(exp_field.shape)
    mask[exp_field.val==0] = 1
    mask_field = ift.Field.from_raw(exp_field.domain, mask)
    mask_operator = ift.MaskOperator(mask_field)
    return mask_operator
#FIXME actually here are pixels (Bad Pixels?) in the middle of the data which are kind of dead which are NOT included in the expfield
#this should be fixed, otherwise we could run into problems with the reconstruction


def convolve_operators(a, b):
    FFT = ift.FFTOperator(a.target)
    convolved = FFT.inverse(FFT(a.real)*FFT(b.real))
    return convolved.real


def convolve_field_operator(field, operator):
    FFT = ift.FFTOperator(operator.target)

    harmonic_field = FFT(field.real)
    fieldOp = ift.DiagonalOperator(harmonic_field.real)

    harmonic_operator = FFT @ operator.real
    convolved = FFT.inverse @ fieldOp @ harmonic_operator
    return convolved.real
