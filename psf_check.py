from jubik0.jwst.psf import build_webb_psf


psf = build_webb_psf(
    camera='nircam',
    filter='f444w',
    center_pixel=(0, 0),
    webbpsf_path='/home/jruestig/Data/jwst/WebbPsf/webbpsf-data_1.2.1/',
    fov_arcsec=4,
    subsample=2)
