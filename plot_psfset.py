import nifty8 as ift
import numpy as np

fileloader = np.load('strainset_psf.npy', allow_pickle=True).item()

psf = fileloader['psf_sim']
sources = fileloader['source']
plt = ift.Plot()
plt_s = ift.Plot()
for i in range(9):
    plt.add(ift.log10(psf[i]))
    plt_s.add(sources[i])

plt.output(nx =3, ny=3, xsize=20, ysize=20, name='s_psf_plot.png')
plt_s.output(nx=3, ny=3, xsize=20, ysize=20, name ="psf_sources.png")
