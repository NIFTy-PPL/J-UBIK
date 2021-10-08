import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

psf1e4 = np.load("psf_1e4.npy", allow_pickle=True).item()
psf1e6 = np.load("psf_1e6.npy", allow_pickle=True).item()
psf1e7 = np.load("psf_1e7.npy", allow_pickle=True).item()

psf1 = psf1e4.val[:, :, 3]
psf1_coll = np.sum(psf1, axis=0)
psf1_norm = np.sum(psf1)
psf1_normed = psf1_coll / psf1_norm

psf2 = psf1e6.val[:, :, 3]
psf2_coll = np.sum(psf2, axis=0)
psf2_norm = np.sum(psf2)
psf2_normed = psf2_coll / psf2_norm

psf3 = psf1e7.val[:, :, 3]
psf3_coll = np.sum(psf3, axis=0)
psf3_norm = np.sum(psf3)
psf3_normed = psf3_coll / psf3_norm

fig, ax = plt.subplots()
ax.plot(psf1_normed[380:460])
ax.plot(psf2_normed[380:460])
ax.plot(psf3_normed[380:460])
ax.set_yscale('log')

print(psf1_norm)
print(psf2_norm)

print(np.sum(psf1_normed[410:430]))
fig.savefig('psf_count_comp.png')

##########
psf3_1 = psf1e7.val[:, :, 0]
psf3_1_coll = np.sum(psf3_1, axis=0)
psf3_1_norm = np.sum(psf3_1)
psf3_1_normed = psf3_1_coll / psf3_1_norm

psf3_2 = psf1e7.val[:, :, 1]
psf3_2_coll = np.sum(psf3_2, axis=0)
psf3_2_norm = np.sum(psf3_2)
psf3_2_normed = psf3_2_coll / psf3_2_norm

psf3_3 = psf1e7.val[:, :, 2]
psf3_3_coll = np.sum(psf3_3, axis=0)
psf3_3_norm = np.sum(psf3_3)
psf3_3_normed = psf3_3_coll / psf3_3_norm

fig, ax = plt.subplots()
ax.plot(psf3_1_normed[380:460], label="low")
ax.plot(psf3_2_normed[380:460], label="med")
ax.plot(psf3_3_normed[380:460], label="high")
ax.legend()
ax.set_yscale('log')

fig.savefig('psf_freq_comp.png')

# Flux check

psf1e6 = np.load("psf_1e6.npy", allow_pickle=True).item()
psf1e6more = np.load("psf_1e6_moreflux.npy", allow_pickle=True).item()

psf_f = psf1e6.val[:, :, 3]
psf_f_coll = np.sum(psf_f, axis=0)
psf_f_norm = np.sum(psf_f)
psf_f_normed = psf_f_coll / psf_f_norm

psf_mf = psf1e6more.val[:, :, 3]
psf_mf_coll = np.sum(psf_mf, axis=0)
psf_mf_norm = np.sum(psf_mf)
psf_mf_normed = psf_mf_coll / psf_mf_norm

print(psf_f_norm)
print(psf_mf_norm)

fig, ax = plt.subplots()
ax.plot(psf_f_normed[380:460], label="1e-3 flux")
ax.plot(psf_mf_normed[380:460], label="1e-1 flux")
ax.legend()
ax.set_yscale('log')
fig.savefig('psf_flux_comp.png', dpi = 400)
