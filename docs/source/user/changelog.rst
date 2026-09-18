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
through East); the old key raises. ``Grid.array_shape`` names the full
numerical field shape, while ``grid.spatial.shape_xy`` and ``shape_yx`` name
the two spatial views. Plot unrotated sky arrays without ``.T``; use WCSAxes
for rotated grids.

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
