import nifty7 as ift
import numpy as np

fileloader = np.load('trainset_psf.npy', allow_pickle=True).item()

psf = fileloader['psf_sim']
sources = fileloader['source']
plt = ift.Plot()
for i in range(5):
    plt.add(ift.log10(psf[i*10]))
    plt.add(sources[i*10])

plt.output(nx =2, ny=5, xsize=10, ysize=50, name='psf_plot.png')