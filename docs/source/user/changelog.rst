Spatial API migration changelog
===============================

2026-09-18 — unreleased, MR !238
------------------------------------------------------------

MR !238 introduces the public XY / internal YX spatial convention.
See :doc:`spatial-conventions` for the current API contract and
:doc:`canonical-sky-design` for the design rationale.

Use public ``shape=(nx, ny)`` integer pixel counts and unit-bearing
``fov=(fov_x, fov_y)`` angular sizes. Key migrations:

* Replace ``spatial_shape`` with ``shape`` and ``rotation`` with
  ``position_angle`` (astronomical North through East).
* ``Grid.shape`` remains the full numerical field shape.
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
* Plot unrotated sky arrays without ``.T``; use WCSAxes for rotated grids.

2026-09-17 — temporary ``sdim`` compatibility, commit e670b109
-------------------------------------------------------------------------------

Square ``sdim`` inputs are accepted temporarily as ``shape`` with a
``FutureWarning`` and removal date of 2026-12-17. Rectangular ``sdim`` values
and supplying both keys raise. Removal still requires a follow-up code change.
