import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def lspn_psf(filename, energy_bin, projection_axis):
    loaded = np.load(filename, allow_pickle=True).item()
    psf = loaded.val[:,:,energy_bin]
    coll = np.sum(psf, axis=projection_axis)
    norm = np.sum(psf)
    res = coll / norm
    return res

def lsp_psf(filename, energy_bin, projection_axis):
    loaded = np.load(filename, allow_pickle=True).item()
    psf = loaded.val[:,:,energy_bin]
    res = np.sum(psf, axis=projection_axis)
    return res

psf1_normed = lspn_psf("psf_1e4.npy", 3, 0)
psf2_normed = lspn_psf("psf_1e6.npy", 3, 0)
psf3_normed = lspn_psf("psf_1e7.npy", 3, 0)

fig, ax = plt.subplots()
ax.plot(psf1_normed[380:460], label="num_rays = 1e4")
ax.plot(psf2_normed[380:460], label="num_rays = 1e6")
ax.plot(psf3_normed[380:460], label="num_rays = 1e7")
ax.set_yscale('log')
ax.legend()
fig.savefig('psf_count_comp.png', dpi=500)

##########

psf3_1_normed = lspn_psf("psf_1e7.npy", 0 , 0)
psf3_2_normed = lspn_psf("psf_1e7.npy", 1 , 0)
psf3_3_normed = lspn_psf("psf_1e7.npy", 2 , 0)
psf3_4_normed = lspn_psf("psf_1e7.npy", 3 , 0)


fig, ax = plt.subplots()
ax.plot(psf3_1_normed[380:460], label="e_bin = 1")
ax.plot(psf3_2_normed[380:460], label="e_bin = 2")
ax.plot(psf3_3_normed[380:460], label="e_bin = 3")
ax.plot(psf3_4_normed[380:460], label="e_bin = 4")
ax.legend()
ax.set_yscale('log')
fig.savefig('psf_freq_comp.png',dpi =500)

# Flux check


psf_f_normed = lspn_psf("psf_1e6.npy",3,0)
psf_mf_normed = lspn_psf("psf_1e6_moreflux.npy",3,0)
psf_emf_normed = lspn_psf("psf_1e6_evenmoreflux.npy",3,0)
psf_Mf_normed = lspn_psf("psf_1e6_megaflux.npy",3,0)
psf_gf_normed = lspn_psf("psf_1e6_gigaflux.npy",3,0)

fig, ax = plt.subplots()
ax.plot(psf_f_normed[380:460], label="1e-3 flux")
ax.plot(psf_mf_normed[380:460], label="1e-1 flux")
ax.plot(psf_emf_normed[380:460], label="1e1 flux")
ax.plot(psf_Mf_normed[380:460], label="1e3 flux")
ax.plot(psf_gf_normed[380:460], label="1e6 flux")
ax.legend()
ax.set_yscale('log')
fig.savefig('psf_flux_comp.png', dpi = 500)

psf_f = lsp_psf("psf_1e6.npy",3,0)
psf_mf= lsp_psf("psf_1e6_moreflux.npy",3,0)
psf_emf= lsp_psf("psf_1e6_evenmoreflux.npy",3,0)
psf_Mf= lsp_psf("psf_1e6_megaflux.npy",3,0)
psf_gf= lsp_psf("psf_1e6_gigaflux.npy",3,0)

fig, ax = plt.subplots()
ax.plot(psf_f[380:460], label="1e-3 flux")
ax.plot(psf_mf[380:460], label="1e-1 flux")
ax.plot(psf_emf[380:460], label="1e1 flux")
ax.plot(psf_Mf[380:460], label="1e3 flux")
ax.plot(psf_gf[380:460], label="1e6 flux")
ax.legend()
ax.set_yscale('log')
fig.savefig('psf_flux_comp_nn.png', dpi = 500)

#TODO More Flux / check different runs for same configs
