Spatial API migration changelog
===============================

2026-09-18 — unreleased, MR !238
------------------------------------------------------------

MR !238 introduces the public XY / internal YX spatial convention.
Its implementation was consolidated in commit ``e995bc54``; commit
``b2dbb061`` adds the compact independent CASA witness and shared validation.
See :doc:`spatial-conventions` for the current API contract and
:doc:`canonical-sky-design` for the design rationale.

Use public ``shape=(nx, ny)`` integer pixel counts and unit-bearing
``fov=(fov_x, fov_y)`` angular sizes. Migration examples::

   # Before
   Grid.from_shape_and_fov(spatial_shape=(320, 192), fov=fov)
   SkyModel(config).create_sky_model(sdim=320)
   WcsAstropy(center, shape, fov, rotation=angle)

   # After
   Grid.from_shape_and_fov(shape=(320, 192), fov=fov)
   SkyModel(config).create_sky_model(shape=320)
   WcsAstropy(center, shape, fov, position_angle=angle)

The YAML ``rotation`` key becomes ``position_angle`` (astronomical North
through East); the old key raises. ``Grid.shape`` remains the familiar full
numerical field shape, with ``Grid.array_shape`` as an explicit alias.
``grid.spatial.shape_xy`` and ``shape_yx`` name the two spatial views. Plot
unrotated sky arrays without ``.T``; use WCSAxes for rotated grids.

Other spatial interfaces become explicit about their order:

* Replace ambiguous ``grid.spatial.shape`` with ``shape_xy`` for public
  geometry or ``shape_yx`` for NumPy arrays.
* Replace ``grid.spatial.fov`` with ``grid.spatial.geometry.fov_xy`` or
  ``fov_yx``; replace ``distances`` with ``pixel_scales_xy`` or
  ``pixel_scales_yx``.
* Replace ``WcsAstropy.get_xycoords`` with the named
  ``world_to_offsets_xy`` / ``offsets_xy_to_world`` or
  ``world_to_indices_yx`` / ``indices_yx_to_world`` conversions.
* ``index_grid_from_bounding_indices(..., indexing="xy")`` is now
  ``pixel_grid_xy_from_bounding_indices(...)``. The root-exported
  ``world_coordinates_to_index_grid`` helper is replaced by
  ``world_to_indices_yx`` for array indices, or Astropy's
  ``world_to_pixel`` when XY pixel coordinates are required.
* JWST interpolation builders no longer accept a selectable ``indexing``
  convention: coordinate grids are always YX. The linear builder now takes
  the expected ``out_shape`` explicitly.

2026-09-17 — temporary ``sdim`` compatibility, commit e670b109
-------------------------------------------------------------------------------

Square ``sdim`` configurations and ``create_sky_model(sdim=...)`` arguments
are temporarily accepted as ``shape``, with a ``FutureWarning`` naming MR
!238, the migration reference and the removal date, 2026-12-17. Rectangular
``sdim`` values remain errors because the old key did not state an axis
order; supplying both ``sdim`` and ``shape`` is also an error.

``SDIM_REMOVAL_DATE`` and a dated removal TODO live in
``jubik/_deprecation.py``. The date documents the planned removal; it is not
an automatic calendar-triggered runtime switch. Removing the shim needs a
follow-up code change after the compatibility window.
