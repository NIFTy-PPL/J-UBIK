import xubik0 as xu
import nifty8 as ift
import timeit


dir_path= "tm1/bcf/"
fname = ["tm1_2dpsf_190219v05.fits", "tm1_2dpsf_190220v03.fits"]

file = dir_path + fname[0]
obs = xu.eROSITA_PSF(file)

energy = '3000'
pointing_center = (2000, 2000)
lower_radec = (0.,0.)
domain = ift.RGSpace((500,500), distances=(8.,8.))

cparams = {'b':(3,3), 'q':(1,1), 'c':(1,1), 'min_m0':(8,8), 'linear':True}
op = obs.make_psf_op(energy, pointing_center, domain, lower_radec,
                     conv_method='MSC', conv_params=cparams)

c2params = {'npatch': 10, 'margfrac': 0.2}
op2 = obs.make_psf_op(energy, pointing_center, domain, lower_radec, 
                      conv_method='LIN', conv_params=c2params)

rnds = ift.from_random(op.domain)

print('JIT MSC-PSF...')
t0 = timeit.default_timer()
res = op(rnds)
t1 = timeit.default_timer()
print('...done JIT MSC-PSF')
print('Compile time MSC:', t1-t0)
res2 = op2(rnds)

t0 = timeit.default_timer()
res = op(rnds)
t1 = timeit.default_timer()
print('MSC:', t1-t0)
t0 = timeit.default_timer()
res2 = op2(rnds)
t1 = timeit.default_timer()
print('LIN:', t1-t0)

cut = op2._cut
res = cut(res)

pl = ift.Plot()
pl.add(res, title = 'MSC')
pl.add(res2, title = 'LIN')
pl.output(nx=2, ny=1, xsize=16, ysize=8)