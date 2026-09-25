# Test data

`camp_branch_28m_dem.tif`: the Camp Branch (Nantahala National Forest, NC) DEM at 28.158 m,
74 x 82 cells, EPSG:32617, clipped to the watershed (no-data outside),
3.44 km2 in 4334 cells; the USGS 1 arc-second tile n36w084 reprojected and
clipped by the DHSVM_Stream_Toolkit (`clip.py`). It is the grid of the
Toolkit's Tier E regression fixture (sha256 181539c9..) and drives the
real-data tests of `terrain.demFromRaster`, `channel_initiation` and the
outlet invariants: 266 channel cells and 30 segments at a support area of
47571.5 m2, a single outlet at cell (68, 68), and a constant-drop
objective of 120 cells (0.0951 km2).
