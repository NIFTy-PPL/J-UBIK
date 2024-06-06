from jubik0.jwst.psf.build_psf import build_webb_psf


psf = build_webb_psf(
    camera='nircam',
    filter='f444w',
    center_pixel=(0, 0),
    webbpsf_path='/home/jruestig/Data/JWST/WebbPsf/webbpsf-data_1.2.1/',
    fov_pixels=12,
    # fov_arcsec=4,
    subsample=2)
