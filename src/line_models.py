import nifty.re as jft
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class LineParameters:
    """
    Container for specifying line parameter priors.

    Each parameter is described by a mean, a standard deviation, 
    and a prior distribution type. This class ensures that means 
    and standard deviations are aligned in shape.

    Parameters
    ----------
    means : jnp.ndarray
        Array of prior means for the parameters.
    stds : jnp.ndarray
        Array of prior standard deviations for the parameters.
        Must have the same shape as `means`.
    prior_type : str
        Type of prior distribution to assume 
        (e.g., ``"normal"`` or ``"lognormal"``).

    Examples
    --------
    Create a prior specification for three parameters:

    >>> params = LineParameters(
    ...     means=jnp.array([1.0, 2.0, 3.0]),
    ...     stds=jnp.array([0.1, 0.2, 0.3]),
    ...     prior_type="normal"
    ... )
    >>> len(params)
    3
    >>> print(params)
    Line 1: 1.0 ± 0.1
    Line 2: 2.0 ± 0.2
    Line 3: 3.0 ± 0.3
    """
    
    means: jnp.ndarray
    stds: jnp.ndarray
    prior_type: str

    def __post_init__(self):
        if not(self.means.shape == self.stds.shape):
            raise ValueError("Means and Stds need to have the same shape")
        
    def __str__(self):
        return "\n".join(f"Line {i+1}: {mean} ± {std}" for i, (mean,std) in enumerate(zip(self.means,self.stds)))
    
    def __len__(self):
        return len(self.means)
        

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


class GaussianPeaks(jft.Model):
    """
    Model for fitting a sum of Gaussian peaks on a fixed grid. 
    Each peak is represented as:

    .. math::

        f(x | c, w, h) = h \cdot \exp\!\left(-\frac{(x - c)^2}{2w^2}\right)

    where :math:`c` is the center, :math:`w` the width, and :math:`h` the height.
    

    Parameters
    ----------
    grid : jnp.ndarray
        1D grid over which peaks are evaluated.
    centers_param : LineParameters
        Prior specification for peak centers.
    widths_param : LineParameters
        Prior specification for peak widths (must be strictly positive).
        Must have the same length as centers_param.
    heights_param : LineParameters
        Prior specification for peak heights.
        Must have the same length as centers_param.
    prefix : str, optional
        Prefix for peak model instance.
    normalized_single_peaks: boolean, optional
        Sets if all peaks should be numerically normalized by the composite trapezoid rule

    Notes
    -----
    If peaks are normalized, the heights are equal to the maxima of the correspoonding peak.
    If peaks are normalized the true height parameter can be regained by integrating over a single peak.
    """
    def __init__(
            self,
            grid: jnp.ndarray,
            centers_param: LineParameters,
            widths_param: LineParameters,
            heights_param: LineParameters,
            prefix: str = None,
            normalized_single_peaks: bool = True,
            ):
        
        if widths_param.prior_type != "lognormal":
            raise ValueError("Peak widths have to be strictly positive. Select 'lognormal' as prior_type.")
        
        prefix = "" if prefix is None else f"{prefix}_"

        self._c = prepare_line_prior(centers_param, name=f"{prefix}gaussian_peak_centers")
        self._w = prepare_line_prior(widths_param, name=f"{prefix}gaussian_peak_widths")
        self._h = prepare_line_prior(heights_param, name=f"{prefix}gaussian_peak_heights")

        self._grid = grid

        if normalized_single_peaks:
            self._peak = self._normalized_gaussian_profile
        else:
            self._peak = self._gaussian_profile
            
        super().__init__(init=self._c.init | self._w.init | self._h.init)

    def _gaussian_profile(self,center,width):
        return jnp.exp(-(1/2)*((self._grid[None,:]-center)/width)**2)
    
    def _normalized_gaussian_profile(self,center,width):
        gaussian_profile = self._gaussian_profile(center,width)
        norm = jnp.trapezoid(gaussian_profile,self._grid,axis=1)
        norm = jnp.maximum(norm, jnp.finfo(gaussian_profile.dtype).tiny)
        return gaussian_profile/norm[:,None]
    
    def centers(self,x):
        return self._c(x)
    
    def widths(self,x):
        return self._w(x)
    
    def heights(self,x):
        return self._h(x)
    
    def single_peaks(self,x):
        # Calculate array of single peaks
        _c = self.centers(x)[:,None]
        _w = self.widths(x)[:,None]
        _h = self.heights(x)[:,None]
        _peaks = self._peak(_c,_w)
        return _h*_peaks

    def __call__(self,x):
        # Gets array of single peaks and sums them up
        return jnp.sum(self.single_peaks(x),axis=0)


class LorentzianPeaks(jft.Model):
    """
    Model for fitting a sum of Lorentzian peaks on a fixed grid.
    Each peak is represented as:

    .. math::

        f(x | c, w, h) = \frac{h}{1 + (\frac{x - c}{w})^2}

    where :math:`c` is the center, :math:`w` the width (half-width at half-maximum),
    and :math:`h` the height.

    Parameters
    ----------
    grid : jnp.ndarray
        1D grid over which peaks are evaluated.
    centers_param : LineParameters
        Prior specification for peak centers.
    widths_param : LineParameters
        Prior specification for peak widths (must be strictly positive).
        Must have the same length as centers_param.
    heights_param : LineParameters
        Prior specification for peak heights.
        Must have the same length as centers_param.
    prefix : str, optional
        Prefix for peak model instance.
    normalized_single_peaks: boolean, optional
        Sets if all peaks should be numerically normalized by the composite trapezoid rule

    Notes
    -----
    If peaks are normalized, the heights are equal to the maxima of the correspoonding peak.
    If peaks are normalized the true height parameter can be regained by integrating over a single peak.
    """
    def __init__(
            self,
            grid: jnp.ndarray,
            centers_param: LineParameters,
            widths_param: LineParameters,
            heights_param: LineParameters,
            prefix: str = None,
            normalized_single_peaks: bool = True,
            ):
        
        if widths_param.prior_type != "lognormal":
            raise ValueError("Peak widths have to be strictly positive. Select 'lognormal' as prior_type.")
        
        prefix = "" if prefix is None else f"{prefix}_"

        self._c = prepare_line_prior(centers_param,name=f"{prefix}lorentzian_centers")
        self._w = prepare_line_prior(widths_param,name=f"{prefix}lorentzian_widths")
        self._h = prepare_line_prior(heights_param,name=f"{prefix}lorentzian_heights")

        self._grid = grid

        if normalized_single_peaks:
            self._peak = self._normalized_lorentzian_profile
        else:
            self._peak = self._lorentzian_profile

        super().__init__(init=self._c.init | self._w.init | self._h.init)

    def _lorentzian_profile(self,center,width):
        inv_profile = (1 + ((self._grid[None,:]-center)/width)**2)
        return 1/inv_profile
    
    def _normalized_lorentzian_profile(self,center,width):
        lorentzian_profile = self._lorentzian_profile(center,width)
        norm = jnp.trapezoid(lorentzian_profile,self._grid,axis=1)
        norm = jnp.maximum(norm, jnp.finfo(lorentzian_profile.dtype).tiny)
        return lorentzian_profile/norm[:,None]
    
    def centers(self,x):
        return self._c(x)
    
    def widths(self,x):
        return self._w(x)
    
    def heights(self,x):
        return self._h(x)
    
    def single_peaks(self,x):
        # Calculate array of single peaks
        _c = self.centers(x)[:,None]
        _w = self.widths(x)[:,None]
        _h = self.heights(x)[:,None]
        _peaks = self._peak(_c,_w)
        return _h*_peaks

    def __call__(self,x):
        # Gets array of single peaks and sums them up
        return(jnp.sum(self.single_peaks(x),axis=0))
     
    
class VoigtPeaks(jft.Model):
    """
    Model for fitting a sum of Voigt peaks on a fixed uniform grid.
    Each Voigt peak is represented as the convolution of a Gaussian and
    Lorentzian profile:

    .. math::

        f(x \mid c, \sigma, \gamma, h) = h \cdot V(x - c; \sigma, \gamma)

    where :math:`c` is the center, :math:`\sigma` is the Gaussian width
    (standard deviation), :math:`\gamma` is the Lorentzian width
    (half-width at half-maximum), and :math:`h` the height. The Voigt
    profile :math:`V` is normalized such that its maximum value is 1.

    Parameters
    ----------
    grid : jnp.ndarray
        1D grid over which peaks are evaluated.
    centers_param : LineParameters
        Prior specification for peak centers.
    gaussian_widths_param : LineParameters
        Prior specification for Gaussian widths (must be strictly positive).
        Must have the same length as centers_param.
    lorentzian_widths_param : LineParameters
        Prior specification for Lorentzian widths (must be strictly positive).
        Must have the same length as centers_param.
    heights_param : LineParameters
        Prior specification for peak heights.
        Must have the same length as centers_param.
    prefix : str, optional
        Prefix for peak model instance.
    normalized_single_peaks: boolean, optional
        Sets if all peaks should be numerically normalized by the composite trapezoid rule

    Notes
    -----
    If peaks are normalized, the heights are equal to the maxima of the correspoonding peak.
    If peaks are normalized the true height parameter can be regained by integrating over a single peak.
    """
    def __init__(
            self,
            grid: jnp.ndarray,
            centers_param: LineParameters,
            gaussian_widths_param: LineParameters,
            lorentzian_widths_param: LineParameters,
            heights_param: LineParameters,
            prefix: str = None,
            normalized_single_peaks: bool = True,
    ):
       
        if (gaussian_widths_param.prior_type != "lognormal") or (lorentzian_widths_param.prior_type != "lognormal"):
            raise ValueError("Peak widths have to be strictly positive. Select 'lognormal' as prior_type for both the lorentzian and gassian widths.")
        
        prefix = "" if prefix is None else f"{prefix}_"


        self._c = prepare_line_prior(centers_param,name=f"{prefix}voigt_centers")
        self._wl = prepare_line_prior(lorentzian_widths_param,name=f"{prefix}voigt_widths_lorentian")
        self._wg = prepare_line_prior(gaussian_widths_param,name=f"{prefix}voigt_widths_gaussian")
        self._h = prepare_line_prior(heights_param,name=f"{prefix}voigt_heights")

        self._grid = grid  

        if normalized_single_peaks:
            self._peak = self._normalized_voigt_profile
        else:
            self._peak = self._voigt_profile

        super().__init__(init=self._c.init | self._wl.init | self._wg.init | self._h.init)

    def _voigt_profile(self, center, width_gauss, width_lorentz):
        N = len(self._grid)
        dg = self._grid[1] - self._grid[0]

        df = 2 * jnp.pi / (N * dg)
        f = (jnp.arange(N) - N//2) * df
        f = f[None,:]

        h_profile = jnp.exp(1j*center*f - (width_gauss*f)**2/2 - width_lorentz*jnp.abs(f))
        h_profile = jnp.fft.ifftshift(h_profile,axis=1)

        profile = jnp.fft.fft(h_profile,axis=1)
        profile = jnp.fft.fftshift(profile,axis=1)/(N*dg)
        profile = jnp.real(profile)

        return profile
    
    def _normalized_voigt_profile(self, center, width_gauss, width_lorentz):
        voigt_profile = self._voigt_profile( center, width_gauss, width_lorentz)
        norm = jnp.trapezoid(voigt_profile,self._grid,axis=1)
        norm = jnp.maximum(norm, jnp.finfo(voigt_profile.dtype).tiny)
        return voigt_profile/norm[:,None]

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
        _c = self.centers(x)[:,None]
        _wg = self.gaussian_widths(x)[:,None]
        _wl = self.lorentzian_widths(x)[:,None]
        _h = self.heights(x)[:,None]
        _peaks = self._peak(_c,_wg,_wl)

        return _h*_peaks
       
    def __call__(self,x):
        # Gets array of single peaks and sums them up
        return(jnp.sum(self.single_peaks(x),axis=0))  