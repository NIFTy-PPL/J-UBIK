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

The former ``sdim`` and grid ``rotation`` keys are rejected with migration
errors; there are no compatibility aliases.

Python callers must make the same clean break. In particular::

   # Before
   Grid.from_shape_and_fov(spatial_shape=(320, 192), fov=fov)
   SkyModel(config).create_sky_model(sdim=(320, 192))

   # After: all public shapes are (nx, ny)
   Grid.from_shape_and_fov(shape=(320, 192), fov=fov)
   SkyModel(config).create_sky_model(shape=(320, 192))

Likewise, ``WcsAstropy(..., rotation=angle)`` becomes
``WcsAstropy(..., position_angle=angle)``.

Internal arrays and explicit metadata
-------------------------------------

A field has trailing shape ``(..., ny, nx)``. At zero position angle, rows
increase North and columns increase West, so East is toward decreasing column
indices. Metadata names state their order explicitly:

===================  ===================
Public XY            Internal NumPy YX
===================  ===================
``shape_xy``         ``shape_yx``
``fov_xy``           ``fov_yx``
``pixel_scales_xy``  ``pixel_scales_yx``
===================  ===================

``Grid.array_shape`` is the full numerical field shape. The ambiguous former
``Grid.shape`` and spatial tuple properties are intentionally absent.

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

Instrument boundaries
---------------------

FITS arrays remain in YX order. The Resolve radio adapter converts the J-UBIK
YX field to its backend-native layout with a pure transpose and never
conjugates it. JWST conversions use explicit YX index methods. Chandra and
eROSITA consume the new ``shape`` key but currently require square spatial
grids and reject rectangular values explicitly.
