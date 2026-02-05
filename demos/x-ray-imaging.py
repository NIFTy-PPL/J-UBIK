# %% [markdown]
# # Imaging of SNR G299.2-2.9 with Chandra
#
# In this demo we are going to image the Super Nova remnant G299.2-2.9 from a chandra observation (id: 11101). Detailed modeling is quite complex, so we will perform a simplified version here.
#
# Before running this demo `J-UBIK` needs to be installed properly, see README for help. After that install [ciao & marx](https://cxc.cfa.harvard.edu/ciao/download/conda.html). We recommend installation of both via conda / conda-forge

# %% [markdown]
# ## Loading data and response
#
# In order to model Chandra and its instrument response, we require details of the instrument configuration at the time the data was collected. 
# Typically, we obtain this information from [`Ciao`](https://cxc.cfa.harvard.edu/ciao/), a software package for Chandra data analysis. 
#
# First Download the data via [Chaser](https://cda.harvard.edu/chaser/) or by running 
#
# ```bash
#     $ download_chandra_obsid 11101
# ```
# in the terminal

# %%
from os import makedirs
import jax
from jax import random

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import nifty.re as jft
from jax.tree_util import Partial

import jubik as ju

# RNG configuration
seed = 42
key = random.PRNGKey(seed)

# %% [markdown]
# ## Set up the scene
# In the first step we use the `ChandraObservationInformation`-Interface to bin the data, get the exposure and simulate the point spread function (PSF). First, we collect the information about the needed filepaths in a python-dictionary.
# We also need to set the pixelization in space and energy, as well as the field of view (fov) and the energy-ranges. In this simple case we are just imaging one very broad energy band.

# %%
obs11101 = {
  "obsID": 11101,
  "data_location": "../data/11101/",
  "event_file": "primary/acisf11101N003_evt2.fits",
  "aspect_sol": "primary/pcadf11101_000N001_asol1.fits",
  "bpix_file": "primary/acisf11101_000N003_bpix1.fits",
  "mask_file": "secondary/acisf11101_000N003_msk1.fits",
  "instrument": "ACIS-I",
}

fov = 960 # in arcsecs, equal 16 arcmin
half_fov_arcmin = fov/2/60 # for plotting

npix_s = 512
npix_e = 1
elim = [0.5, 7] # in keV
energy_ranges = (0.5, 7.0) # in keV
pixel_size = (fov/npix_s)**2 # 


outroot = "ChandraSNRdemo"
makedirs(outroot, exist_ok=True)

# %% [markdown]
# ## Get Data, Exposure and PSF
# In the next step we use the interface to retrieve the binned data, the exposure and the PSF and plot it.

# %%
info = ju.ChandraObservationInformation(obs11101,
                                        npix_s=npix_s,
                                        npix_e=npix_e,
                                        fov=fov,
                                        elim=elim,
                                        energy_ranges=energy_ranges)


# %%
# retrieve data from observation
data = info.get_data("data_11101.fits")
data = data[:,:,0]

# compute the exposure map
exposure = info.get_exposure("exposure_11101.fits")
exposure = exposure[:,:,0]

# compute the on axis psf
psf = info.get_psf_fromsim((info.obsInfo["aim_ra"],
                            info.obsInfo["aim_dec"]),
                            "psf_11101.fits",
                            num_rays=1e7)
psf = psf[:,:,0]

# %%
plt.imshow(data, origin="lower", interpolation="none", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin], norm="log")
plt.colorbar(label="photon counts")
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.savefig(outroot+"/data.png")

# %% [markdown]
# ![](ChandraSNRdemo/data.png)

# %%
plt.imshow(exposure, origin="lower", interpolation="none", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin], norm="log")
plt.colorbar(label=r"$\mathrm{s}\times \mathrm{cm}^2$")
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.savefig(outroot+"/exposure.png")

# %% [markdown]
# ![](ChandraSNRdemo/exposure.png)

# %%
plt.imshow(psf, origin="lower", interpolation="none", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin], norm="log")
plt.colorbar(label="photon counts")
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.savefig(outroot+"/psf.png")

# %% [markdown]
# ![](ChandraSNRdemo/psf.png)

# %% [markdown]
# ## Diffuse Flux Model
# In the next step, we model the sky brightness with the `NIFTy.re` correlated field model, using the information about the observation (FOV, pixelization). More Information about this is provided in the [NIFTy.re tutorials](https://ift.pages.mpcdf.de/nifty/user/notebooks_re/4_correlated_field_model.html#). 
#   
# Since flux is a positive quantity, we are going to build a lognormal correlated field model to image the SNR. This is going to model the diffuse flux we see in the data.

# %%
grid = (npix_s, npix_s) # Grid shape
cf_zm = dict(offset_mean=-16.0, offset_std=(1, 1)) 
cf_fl = dict(
    fluctuations=(1.0, 0.5),
    loglogavgslope=(-2.7, 0.5),
    flexibility=None,
    asperity=None,
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    grid, distances= fov / grid[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
log_diffuse = cfm.finalize()
diffuse = lambda x: jnp.exp(log_diffuse(x))

# %% [markdown]
# ## Sample from the Diffuse Flux Prior
# Generally models are built in a way, that the priors are standardized normal distributions. Therefore, in order to draw samples from this prior, we draw standard normal distributed numbers, here called $\xi$, and insert them in the diffuse flux model we just set up.

# %%
key, subkey = random.split(key)
xi = jft.random_like(subkey, log_diffuse.domain)
diffuse_prior_sample = diffuse(xi)

# %%
plt.imshow(diffuse_prior_sample, origin="lower", norm="log", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin], interpolation="none")
plt.colorbar(label=r"$\mathrm{s}^{-1}\mathrm{cm}^{-2}$")
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.savefig(outroot+"/diffuse_sample.png")

# %% [markdown]
# ![](ChandraSNRdemo/diffuse_sample.png)

# %% [markdown]
# ## Point Source Model
# To model point sources, we use the [Inverse Gamma Prior](https://en.wikipedia.org/wiki/Inverse-gamma_distribution). We use the implementation [NIFTy.re.InvGammaPrior](https://ift.pages.mpcdf.de/nifty/mod/nifty.re.prior.html#nifty.re.prior.InvGammaPrior).
# This distribution has two parameters, alpha and scale.

# %%
alpha = 1.005
scale = 1e-9
points = jft.InvGammaPrior(alpha, scale, shape=grid, name="points")

# %% [markdown]
# ## Sample from the Point Source Flux Prior
# In the same manner, as in the diffuse model, we can draw a sample for the point source prior.

# %%
key, subkey = random.split(key)
xi = jft.random_like(subkey, points.domain)
points_prior = points(xi)

# %%
plt.imshow(points_prior, norm="log", vmin=1e-8, origin="lower", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin], interpolation="none")
plt.colorbar(label=r"$\mathrm{s}^{-1}\mathrm{cm}^{-2}$")
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.savefig(outroot+"/ps_sample.png")

# %% [markdown]
# ![](ChandraSNRdemo/ps_sample.png)

# %% [markdown]
# ## Sky Brightness Model
# For the inference we wrap the two functions into a `NIFTy.re.Model`. Therfore, we build a callable which adds the results of the two callables, and merge the domains.

# %%
sky = jft.Model(
    call=lambda x: diffuse(x)+ points(x), domain=log_diffuse.domain | points.domain, white_init=True
)

# %%
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, sky.domain)
prior_sky = sky(pos_truth)

# %%
plt.imshow(prior_sky, origin="lower", norm="log", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin], interpolation="none")
plt.colorbar(label=r"$\mathrm{s}^{-1}\mathrm{cm}^{-2}$")
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.savefig(outroot+"/sky_sample.png")


# %% [markdown]
# ![](ChandraSNRdemo/sky_sample.png)

# %% [markdown]
# ## The Instrument Response of Chandra
#
# The full instrument response $R$ for Chandra is basically:
#
# $$R = M \circ E \circ P$$
#
# $M$ is the mask, $E$ the exposure, $P$ the PSF.
# For this example we assume that it is enough to consider these linear effects, and that pile-up can be neglected. Also we assume that the PSF, $P$, is spatially invariant.

# %%
def chandra_response(exposure, psf):
    """Convenience function to retrieve the instrument response 
    for this observation

    Parameters:
    -----------
    exposure: np.array
    psf: np.array

    Returns:
    -------
    signal_response: callable
    """
    # prep psf for convolution (normalize and shift)
    integral = psf.sum()*pixel_size
    psf = psf / integral
    psf_shift = jnp.fft.fftshift(psf)
    
    def convolve(signal):
        """FFT convolution code"""
        psf_k = jnp.fft.fftn(psf_shift)
        signal_k = jnp.fft.fftn(signal)
        conv = jnp.fft.ifftn(signal_k*psf_k)
        return conv.real

    def expose(signal):
        exposed = exposure*signal
        return exposed

    def mask(signal):
        """Mask all unexposed parts of the signal."""
        flag = exposure!=0
        return signal[flag]

    # Function to backproject the data to image domain
    mask_adj_temp = jax.linear_transpose(mask, jnp.ones(grid))
    mask_adj = lambda x: mask_adj_temp(x)[0]
    # All parts of the response in a dict
  
    # Full response
    signal_response = lambda x: mask(expose(convolve(x)))
    dct = {"conv": convolve,
           "exposure": expose,
           "mask": mask,
           "mask_adj": mask_adj,
          "signal_response": signal_response,}
    
    return dct


# %%
instrument_dct = chandra_response(exposure, psf)
R = instrument_dct["signal_response"]
M_adj = instrument_dct["mask_adj"]
M = instrument_dct["mask"]

# %% [markdown]
# ## Model of signal response
# In order to use the full functionality of `NIFTy.re` we are also wrapping the full call, Response applied to the Sky, in a Model. Then we want to visualize the effect of the instrument response representation of Chandra to our sky sample.

# %%
signal_response = jft.Model(call=lambda x: R(sky(x)), domain=sky.domain, white_init=True
)

# %%
signal_response_sample = M_adj(R(prior_sky)) # Since the data space has not structure after masking, we need to backproject to the sky, with M_adj.

# %%
plt.imshow(signal_response_sample, origin="lower", norm="log", vmin=0.1, interpolation="none", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin])
plt.colorbar(label="mean expected counts")
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.savefig(outroot+"/signal_response_sample.png")

# %% [markdown]
# ![](ChandraSNRdemo/signal_response_sample.png)

# %% [markdown]
# ## Likelihood
# Since the data consists of photon counts we'll use a Poissonian likelihood `jft.Poissonian`. The Poissonian Hamiltonian (the negative log likelihood) is
# $$\log\mathcal{P}(d|s)=  \sum \lambda - d\log\lambda$$
#
# We are going mask all pixels with 0 exposure, since 0 exposure causes $\lambda$ to be 0, which leads to `NaN` values in the likelihood.

# %%
m_data = M(data)

# %%
lh = jft.Poissonian(m_data).amend(signal_response)


# %% [markdown]
# ## Inference
# We use NIFTy.re.optimize_kl to do the inference. Here more information about [VI inference schemes](https://ift.pages.mpcdf.de/nifty/user/approximate_inference.html), used in this, and [Information Field Theory](https://ift.pages.mpcdf.de/nifty/user/ift.html).
# OptimizeKL, used later, can use a so-called callback function, which is called after every KL-minimzation round. This can be used to plot intermediate result. Here a helpfull plotting routine.

# %%
def _imshow(figure, field, ax, title, cbarlabel, vmin=1e-8, vmax=None, norm=None, cmap="viridis"):
    im0 = ax.imshow(
        field,
        origin="lower",
        extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin],
        vmin=vmin,
        cmap=cmap,
        vmax=vmax,
        norm=norm,
        interpolation="none",
    )
    figure.colorbar(im0, ax=ax, label=cbarlabel)
    ax.set_xlabel(r"x in [arcmin]")
    ax.set_ylabel(r"y in [arcmin]")
    ax.set_title(title)
    
def plotting_callback(samples, opt_state):
    m_data = M(data)
    def nwr(sky):
        sr = R(sky)
        nwr=(m_data-sr)/jnp.sqrt(sr)
        nwr_image = M_adj(nwr)
        return nwr_image
        
    sky_mean, sky_std = jft.mean_and_std(tuple(sky(s) for s in samples))
    nwr_mean = jft.mean(tuple(nwr(sky(s)) for s in samples))
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 13))
    axs = axs.flatten()
    
    _imshow(fig, sky_mean, 
            axs[0],
            "Sky Brightness Mean",
            norm="log", 
            cbarlabel="$\mathrm{s}^{-1}\mathrm{cm}^{-2}$",
            cmap="viridis",
            )
    _imshow(fig, sky_std, 
            axs[1],
            "Sky Brightness Std",
            norm="log",
            cbarlabel="$\mathrm{s}^{-1}\mathrm{cm}^{-2}$",
            cmap="viridis"
            )
    _imshow(fig, data/exposure, 
            axs[2], 
            "Exposure Corrected Data",
            norm="log",
            cbarlabel="$\mathrm{s}^{-1}\mathrm{cm}^{-2}$",
            cmap="viridis")
    _imshow(fig,
            nwr_mean,
            axs[3],
            "Noise Weighted Residuals",
            cbarlabel="in units of stds",
            vmin=-5,
            vmax=5,
            cmap="bwr")

    axs[1].yaxis.set_visible(False)
    axs[3].yaxis.set_visible(False)

    fig.tight_layout()
    file_name = outroot+f"/x_ray_imaging_{opt_state.nit}.png"
    plt.savefig(file_name)
    plt.close()


# %%
key, k_i, k_o = random.split(key, 3)
n_vi_iterations = 5
delta = 1e-6
n_samples = 4

samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=20
        )
    ),
    sample_mode="linear_resample",
    callback=plotting_callback,
    odir=outroot,
    resume=False,
)

# %% [markdown]
# ![](ChandraSNRdemo/x_ray_imaging_5.png)

# %% [markdown]
# ## More Plotting
# Since we modeled diffuse and point source emission individually, we can also plot them individually. THerefore, we calculate the posterior mean, or a sample mean, using the inferred samples.
#
# $$\langle f(\xi)\rangle = \frac{1}{N} \Sigma f(\xi)$$
#
#  where $f(\xi)$ is the application of the point source or the diffuse model to the latent variables $\xi$. We apply the Mask and its Adjoint to remove the regions, which are uninformed by data.

# %%
post_diff_sky = jnp.array([M_adj(M(diffuse(s))) for s in samples])

post_diffuse_mean = jnp.mean(post_diff_sky, axis=0)
plt.imshow(post_diffuse_mean, 
           origin="lower",
           norm="log",
           vmin=1e-7,
           extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin],
           interpolation="none")
plt.colorbar(label=r"$\mathrm{s}^{-1}\mathrm{cm}^{-2}$")
plt.xlabel(r"x in [arcmin]")
plt.ylabel(r"y in [arcmin]")
plt.savefig(outroot+"/post_mean_diff.png")


# %% [markdown]
# ![](ChandraSNRdemo/post_mean_diff.png)

# %%
post_point = jnp.array([M_adj(M(points(s))) for s in samples])
post_point_mean = jnp.mean(post_point, axis=0)
plt.imshow(post_point_mean, 
           origin="lower",
           norm="log",
           vmin=1e-7,
           extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin],
           interpolation="none")
plt.colorbar(label=r"$\mathrm{s}^{-1}\mathrm{cm}^{-2}$")
plt.xlabel(r"x in [arcmin]")
plt.ylabel(r"y in [arcmin]")
plt.savefig(outroot+"/post_mean_points.png")

# %% [markdown]
# ![](ChandraSNRdemo/post_mean_points.png)
