Spatial coordinate conventions
==============================

J-UBIK exposes spatial geometry in Cartesian ``(x, y)`` order and stores
numerical fields in NumPy ``(..., y, x)`` order. The conversion is owned by
:class:`jubik.wcs.SpatialGeometry`; callers use named XY or YX views instead of
reversing tuples themselves.

Public geometry
---------------

``shape=(nx, ny)``, ``fov=(fov_x, fov_y)``, and public celestial offsets
``(x_east, y_north)`` all use XY order. Scalar ``shape`` and ``fov`` values are
broadcast to two equal components. Celestial offsets must carry Astropy units.
The astronomical ``position_angle`` is measured from North toward East.

YAML uses the same convention::

   grid:
     shape: [320, 192]
     fov: [48arcsec, 24arcsec]
     position_angle: 0deg
     frame: icrs
     sky_center: {ra: 10deg, dec: -5deg}

For deprecated keys and migration examples, see :doc:`changelog`.

Which RA/Dec frame?
----------------------------------------

``frame`` selects the celestial reference system, independently of XY/YX
array order. The default is ``icrs``. For equatorial configurations, the
other choices are ``fk5`` (default equinox ``J2000.0``) and ``fk4`` (default
equinox ``B1950.0``). An explicit ``equinox`` is accepted only for FK4/FK5;
do not set it for ICRS. For example::

   grid:
     shape: [320, 192]
     fov: [48arcsec, 24arcsec]
     frame: fk5
     equinox: J1990.0
     sky_center: {ra: 10deg, dec: -5deg}
     position_angle: 30deg

The ``sky_center`` values are interpreted in the selected frame/equinox.
``Grid.from_grid_model(GridModel.from_yaml_dict(config["grid"]))`` carries
these settings into ``grid.spatial.coordinate_system`` and the WCS/FITS
``RADESYS`` and ``EQUINOX`` metadata. Read ``grid.spatial.center`` for the
reference sky position. ``position_angle`` changes the grid orientation,
not its celestial reference system. The key is ``frame``, not
``coordinate_frame``.

The direct ``Grid.from_shape_and_fov`` convenience constructor builds an
ICRS WCS; use the config/``GridModel`` path to select FK4/FK5 and an equinox.
The coordinate-system enum also includes Galactic coordinates, but the
current YAML center parser takes equatorial ``ra``/``dec`` fields: a Galactic
``l``/``b`` YAML center is not supported by that path.

Internal arrays and explicit metadata
-------------------------------------

A field has trailing shape ``(..., ny, nx)``. At zero position angle, rows
increase North and columns increase West, so East is toward decreasing column
indices. Metadata names state their order explicitly:

.. list-table:: Physical coordinates versus array indices
   :header-rows: 1

   * - View
     - Order / meaning
     - Example
   * - Public geometry (pixel counts and angular sizes)
     - XY, ``(nx, ny)`` / ``(fov_x, fov_y)``
     - ``shape_xy``, ``fov_xy``, ``pixel_scales_xy``
   * - Public physical offsets (unit-bearing sky angles)
     - XY, ``(East, North)``; not pixel indices
     - ``world_to_offsets_xy``
   * - Internal arrays and pixel indices
     - YX, ``(row, column)``; at zero PA, row grows North,
       column grows West (opposite the positive East offset)
     - ``shape_yx``, ``pixel_scales_yx``, ``world_to_indices_yx``
   * - Full numerical field
     - ``(polarization, time, spectral, y, x)``
     - ``Grid.shape``

At non-zero position angle, use the WCS transform to determine the world
direction of a pixel step; the storage order is still YX. With unequal pixel
scales, the current FITS ``CDELT * PC`` convention can shear the sky grid
rather than rigidly rotate it. The example below uses square pixels.

Coordinate conversion and plotting
----------------------------------

Use ``world_to_offsets_xy`` and ``offsets_xy_to_world`` for unit-bearing East,
North offsets. Use ``world_to_indices_yx`` and ``indices_yx_to_world`` at NumPy
array boundaries. The pixel grid itself lives on ``grid.spatial.geometry``, a
``SpatialGeometry``: read ``n_ra``, ``n_dec``, ``d_ra``, ``d_dec`` there instead
of indexing a shape tuple. To map every pixel to the sky, feed
``np.indices(grid.spatial.shape_yx)`` to ``indices_yx_to_world``; the method
name says which order it expects.

An unrotated field is plotted directly, without a transpose::

   field = np.asarray(model(position))
   plt.imshow(field, origin="lower", extent=grid.spatial.extent())
   plt.xlabel("East offset")
   plt.ylabel("North offset")

``extent()`` returns East-left bounds ``(+half_x, -half_x, -half_y, +half_y)``.
It rejects rotated grids because a rectangular Matplotlib extent cannot encode
a rotated celestial transform; use WCSAxes for those images.

The self-contained ``demos/grids.py`` example draws an actual random sky with
ICRS coordinate labels at position angles 0 and 30 degrees. Both panels
display exactly the same YX array: the celestial grid changes, not the
storage order. Run ``MPLBACKEND=Agg python demos/grids.py`` headlessly, or
omit the backend setting to open the figure interactively.

.. plot:: user/grids.py
   :include-source: false

   Random sky on unrotated and rotated celestial axes. No transpose or
   rectangular ``extent`` is applied; WCSAxes uses the full WCS.

See Astropy's `WCSAxes introduction
<https://docs.astropy.org/en/stable/visualization/wcsaxes/initializing_axes.html>`_
for the plotting interface.

Where does the instrument point?
----------------------------------------

An instrument pointing is a physical sky location (``SkyCoord``), not a
row/column pair. The reconstruction center defines its own WCS reference
position; an instrument may point elsewhere. Convert a pointing through that
WCS before placing it on a sky array. All instruments read the same YX sky:

.. list-table:: Pointing and layout at instrument boundaries
   :header-rows: 1

   * - Instrument
     - Physical pointing / coverage
     - Array boundary
   * - JWST
     - The data's gwcs maps detector pixels to sky coordinates;
       reconstruction WCS maps those to YX sky indices.
     - Interpolation samples the canonical YX sky; detector orientation
       need not align with reconstruction rows and columns.
   * - Resolve (radio)
     - The observation's phase center and beam pointing are sky coordinates;
       the beamer places the beam on the reconstruction WCS.
     - ``canonical_sky_to_visibilities`` transposes YX once to the raw
       gridder's layout; no conjugation or extra sign flip.
   * - Chandra
     - Observation astrometry defines sky coverage relative to the
       reconstruction center.
     - Square-grid response consumes the sky in YX order; rectangles
       are rejected by the config adapter.
   * - eROSITA
     - Observation astrometry and per-module coverage locate the field
       relative to the reconstruction center.
     - Square-grid response consumes the sky in YX order; rectangles
       are rejected by the config adapter.

FITS arrays remain YX; FITS header axis 1 is the column axis and axis 2 is the
row axis. A backend conversion does not redefine the physical pointing or
the shared sky convention. The :doc:`canonical-sky-design` page records
the named seams and the independent tests that pin them.
