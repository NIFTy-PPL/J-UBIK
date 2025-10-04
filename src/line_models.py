import nifty.re as jft
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import vmap
from dataclasses import dataclass

@dataclass(frozen=True)
class LineParameters:
    means: jnp.ndarray
    stds: jnp.ndarray
    prior_type: str

    def __post_init__(self):
        if not(self.means.shape == self.stds.shape):
            raise ValueError("Means and Stds need to have the same shape")
        
    def __str__(self):
        return "\n".join(f"Line {i+1}: {mean} ± {std}" for i, (mean,std) in enumerate(zip(self.means,self.stds)))
        

def prepare_line_prior(line_parameter: LineParameters, name):
    prior_type = line_parameter.prior_type
    means = line_parameter.means
    stds = line_parameter.stds
    shp = means.shape
    
    supported_priors = ["normal","lognormal"]
    if prior_type not in supported_priors:
        raise NotImplementedError("Can only create normal or lognormal prior.")
    
    if prior_type == "normal":
        prior = jft.NormalPrior(mean=means,std=stds,name=name,shape=shp)

    if prior_type == "lognormal":
        prior = jft.LogNormalPrior(mean=means,std=stds,name=name,shape=shp)

    return prior

def gaussian_profile(c,w,h,grid):
    dg = grid[1] - grid[0]
    profile = jnp.exp(-(1/2)*((grid-c)/w)**2)/(jnp.sqrt(2*jnp.pi)*w)
    profile /= jnp.sum(profile*dg)
    return h*profile

def lorentzian_profile(c,w,h,grid):
    dg = grid[1] - grid[0]
    inv_profile = jnp.pi*w*(1 + ((grid-c)/w)**2)
    profile = 1/inv_profile
    profile /= jnp.sum(profile*dg)
    return h*profile

def voigt_profile(c, w_gauss, w_lorentz, h, grid):
    N = len(grid)
    dg = grid[1] - grid[0]

    df = 2 * jnp.pi / (N * dg)
    f = (jnp.arange(N) - N//2) * df

    h_profile = jnp.exp(1j*c*f - (w_gauss*f)**2/2 - w_lorentz*jnp.abs(f))
    h_profile = jnp.fft.ifftshift(h_profile)

    profile = jnp.fft.fft(h_profile)
    profile = jnp.fft.fftshift(profile)/(N*dg)
    profile = jnp.real(profile)
    profile /= jnp.sum(profile*dg)

    return h*profile


class GaussianPeaks(jft.Model):
    def __init__(
            self,
            grid: jnp.ndarray,
            centers_param: LineParameters,
            widths_param: LineParameters,
            heights_param: LineParameters,
            prefix=""
            ):
        
        if widths_param.prior_type != "lognormal":
            raise ValueError("Peak widths have to be strictly positive. Select 'lognormal' as prior_type.")

        self._c = prepare_line_prior(centers_param, name=f"{prefix}_gaussian_peak_centers")
        self._w = prepare_line_prior(widths_param, name=f"{prefix}_gaussian_peak_widths")
        self._h = prepare_line_prior(heights_param, name=f"{prefix}_gaussian_peak_heights")

        self._grid = grid
    
        self._gaussian_profile = Partial(gaussian_profile,grid=self._grid)

        super().__init__(init=self._c.init | self._w.init | self._h.init)

    def __call__(self,x):
        # Gets array of single peaks and sums them up
        return jnp.sum(self.single_peaks(x),axis=0)
    
    def centers(self,x):
        return self._c(x)
    
    def widths(self,x):
        return self._w(x)
    
    def heights(self,x):
        return self._h(x)
    
    def single_peaks(self,x):
        # Calculate array of single peaks
        _c = self.centers(x)
        _w = self.widths(x)
        _h = self.heights(x)
        return vmap(self._gaussian_profile, in_axes=(0,0,0))(_c,_w,_h)
    
class LorentzianPeaks(jft.Model):
    def __init__(
            self,
            grid: jnp.ndarray,
            centers_param: LineParameters,
            widths_param: LineParameters,
            heights_param: LineParameters,
            prefix=""
            ):
        self._c = prepare_line_prior(centers_param,name=f"{prefix}_lorentzian_centers")
        self._w = prepare_line_prior(widths_param,name=f"{prefix}_lorentzian_widths")
        self._h = prepare_line_prior(heights_param,name=f"{prefix}_lorentzian_heights")

        self._grid = grid
    
        self._lorentzian_profile = Partial(lorentzian_profile,grid=self._grid)

        if widths_param.prior_type != "lognormal":
            raise ValueError("Peak widths have to be strictly positive. Select 'lognormal' as prior_type.")

        super().__init__(init=self._c.init | self._w.init | self._h.init)

    def __call__(self,x):
        # Gets array of single peaks and sums them up
        return(jnp.sum(self.single_peaks(x),axis=0))
    
    def centers(self,x):
        return self._c(x)
    
    def widths(self,x):
        return self._w(x)
    
    def heights(self,x):
        return self._h(x)
    
    def single_peaks(self,x):
        # Calculate array of single peaks
        _c = self.centers(x)
        _w = self.widths(x)
        _h = self.heights(x)
        return vmap(self._lorentzian_profile, in_axes=(0,0,0))(_c,_w,_h)
    
class VoigtPeaks(jft.Model):
    def __init__(
            self,
            grid: jnp.ndarray,
            centers_param: LineParameters,
            gaussian_widths_param: LineParameters,
            lorentzian_widths_param: LineParameters,
            heights_param: LineParameters,
            prefix=""
    ):
        self._c = prepare_line_prior(centers_param,name=f"{prefix}_voigt_centers")
        self._wl = prepare_line_prior(lorentzian_widths_param,name=f"{prefix}_voigt_widths_lorentian")
        self._wg = prepare_line_prior(gaussian_widths_param,name=f"{prefix}_voigt_widths_gaussian")
        self._h = prepare_line_prior(heights_param,name=f"{prefix}_voigt_heights")

        self._grid = grid  

        self._voigt_profile = Partial(voigt_profile,grid=self._grid)

        if (gaussian_widths_param.prior_type != "lognormal") or (lorentzian_widths_param.prior_type != "lognormal"):
            raise ValueError("Peak widths have to be strictly positive. Select 'lognormal' as prior_type for both the lorentzian and gassian widths.")

        super().__init__(init=self._c.init | self._wl.init | self._wg.init | self._h.init)

    def __call__(self,x):
        # Gets array of single peaks and sums them up
        return(jnp.sum(self.single_peaks(x),axis=0))
    
    def centers(self,x):
        return self._c(x)
    
    def lorentzian_widths(self,x):
        return self._wl(x)

    def gaussian_widths(self,x):
        return self._wg(x)
    
    def heights(self,x):
        return self._h(x)
    
    def single_peaks(self,x):
        # Calculate array of single peaks
        _c = self.centers(x)
        _wg = self.gaussian_widths(x)
        _wl = self.lorentzian_widths(x)
        _h = self.heights(x)
        return vmap(self._voigt_profile, in_axes=(0,0,0))(_c,_wg,_wl,_h)