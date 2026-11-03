"""Units at the boundary between quantity-carrying skies and radio responses.

The response accepts numerical sky brightness in Jy/sr and integrates over
pixel solid angle to produce visibilities in Jy. Its adjoint output is assigned
the same sky unit by ``dirty_image``. Angular pixel spacing is converted to
radians before entering the response backends. Frequency bins are expressed in
Hz to match Observation's frequency arrays; effective UVW coordinates are
baselines in wavelengths, computed from UVW in metres and frequencies in Hz.

``interferometry_response`` converts grid quantities to these conventions, and
``radio_sky_extractor`` converts sky brightness. ``dirty_image`` attaches the sky
unit before converting to the caller's requested output unit. Observation keeps
bare numerical arrays; it does not carry arbitrary units.

A future unit-carrying Observation refactor (issue #114) must make equality
unit-aware and preserve units across save/load, and update every consumer of
the numerical arrays.
"""

import astropy.units as u

RESPONSE_SKY_UNIT = u.Jy / u.sr
RESPONSE_SPATIAL_UNIT = u.rad
RESPONSE_SPECTRAL_UNIT = u.Hz
