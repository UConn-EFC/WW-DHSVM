"""Tests of the constant-drop channel-initiation analysis and the
bring-your-own-DEM entry, on a synthetic basin and on Camp Branch.

Runs without a DHSVM binary and without the data layer: numpy, pyflwdir,
rasterio and pandas only (scipy if present, for the t-statistic check).
"""
import numpy as np
import pytest

import ww_dhsvm.grid as G
import ww_dhsvm.crs as C
import ww_dhsvm.terrain as T
import ww_dhsvm.streams as S
import ww_dhsvm.channel_initiation as CI

from pathlib import Path

CA_DEM = Path(__file__).resolve().parent / 'data' / 'camp_branch_28m_dem.tif'


# ------------------------------------------------------------ synthetic
@pytest.fixture(scope='module')
def synthetic():
    """A tilted, roughened surface draining to the south-east corner,
    the basin being everything but a border ring (one clean outlet)."""
    ny, nx, cell = 60, 80, 150.0
    grid = G.ModelGrid(ny, nx, cell, 700000.0, 4600000.0, C.from_epsg(32618))
    yy, xx = np.mgrid[0:ny, 0:nx]
    dem = (900.0 - 4.0 * yy - 2.5 * xx
           + 35 * np.sin(xx / 7.0) * np.cos(yy / 9.0)).astype('float32')
    mask = np.zeros((ny, nx), 'uint8')
    mask[2:-2, 2:-2] = 1
    grid.mask = mask
    terrain = T.conditionDEM(dem, grid)
    return grid, terrain


def test_welch_t_matches_scipy():
    scipy_stats = pytest.importorskip('scipy.stats')
    rng = np.random.default_rng(1)
    a = rng.normal(10, 3, 15)
    b = rng.normal(12, 5, 9)
    t, _ = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert abs(CI.welchT(a, b) - t) < 1e-12
    assert np.isnan(CI.welchT(a[:1], b))


def test_strahler_order_is_recomputed_for_each_mask(synthetic):
    grid, terrain = synthetic
    flw, acc = terrain['flowdir'], terrain['flowacc']
    dense = (acc >= 20) & (grid.mask != 0)
    sparse = (acc >= 400) & (grid.mask != 0)
    so_dense = CI.strahlerOrder(flw, dense)
    so_sparse = CI.strahlerOrder(flw, sparse)
    assert so_dense.max() > so_sparse.max() >= 1
    assert int(so_sparse[~sparse].max()) == 0
    # the same answers whatever the order of the calls
    assert np.array_equal(CI.strahlerOrder(flw, dense), so_dense)
    assert np.array_equal(CI.strahlerOrder(flw, sparse), so_sparse)


def test_first_sustained_band():
    rows = [dict(cells=c, passes=p) for c, p in
            [(10, False), (20, True), (30, False), (40, True), (50, True),
             (60, True), (70, False), (80, True), (90, True), (100, True)]]
    obj, band = CI.firstSustainedBand(rows, 3)
    assert obj['cells'] == 40 and [r['cells'] for r in band] == [40, 50, 60]
    obj1, band1 = CI.firstSustainedBand(rows, 1)
    assert obj1['cells'] == 20 and len(band1) == 1
    assert CI.firstSustainedBand(rows, 4) == (None, [])
    assert CI.firstSustainedBand([], 3) == (None, [])


def test_drop_analysis_table_on_the_synthetic_basin(synthetic):
    grid, terrain = synthetic
    res = CI.dropAnalysis(terrain, grid, tmin_cells=10, tmax_cells=400, step_cells=30)
    tab = res['table']
    assert list(tab['cells']) == list(range(10, 401, 30))
    assert (tab['n_stream_cells'].diff().dropna() <= 0).all()
    assert (tab['max_order'].diff().dropna() <= 0).all()
    assert (tab['n_streams'] == tab['n_order1'] + tab['n_higher']).all()
    assert (tab['drainage_density'] > 0).all()
    assert tab['passes'].dtype == bool
    if res['objective_cells'] is not None:
        lo, hi = res['band_cells']
        assert lo == res['objective_cells'] <= hi
        assert abs(res['objective_km2'] - lo * grid.cell_area / 1e6) < 1e-12


def test_extract_network_with_drop_threshold(synthetic):
    grid, terrain = synthetic
    res = CI.dropAnalysis(terrain, grid, tmin_cells=10, tmax_cells=400, step_cells=30,
                          min_band=1)
    if res['objective_km2'] is None:
        pytest.skip('no passing threshold on the synthetic surface')
    kw = dict(tmin_cells=10, tmax_cells=400, step_cells=30, min_band=1)
    net_drop = S.extractNetwork(terrain, grid, channel_threshold_km2='drop', drop_kwargs=kw)
    net_num = S.extractNetwork(terrain, grid, channel_threshold_km2=res['objective_km2'])
    assert net_drop['channel_threshold_km2'] == res['objective_km2']
    assert net_drop['drop_analysis']['objective_cells'] == res['objective_cells']
    assert np.array_equal(net_drop['channel_mask'], net_num['channel_mask'])
    assert len(net_drop['segments']) == len(net_num['segments'])
    with pytest.raises(ValueError):
        S.extractNetwork(terrain, grid, channel_threshold_km2='auto')


# ---------------------------------------------------------- Camp Branch
@pytest.fixture(scope='module')
def camp_branch():
    grid, dem = T.demFromRaster(str(CA_DEM))
    terrain = T.conditionDEM(dem.astype('float32'), grid)
    return grid, dem, terrain


def test_dem_from_raster_camp_branch(camp_branch):
    grid, dem, terrain = camp_branch
    assert grid.shape == (74, 82)
    assert abs(grid.cellsize - 28.15774) < 1e-4
    assert int(C.to_epsg(grid.crs)) == 32617
    assert int(grid.mask.sum()) == 4334
    assert abs(grid.basin_area_km2 - 3.436) < 0.001
    assert not np.isnan(dem).any()
    inside = grid.mask != 0
    assert abs(float(dem[inside].min()) - 826.8) < 0.1
    assert abs(float(dem[inside].max()) - 1625.2) < 0.1


def test_grid_from_raster_rejects_a_misaligned_grid(camp_branch, tmp_path):
    grid, dem, terrain = camp_branch
    shifted = G.ModelGrid(grid.nrows, grid.ncols, grid.cellsize,
                          grid.xllcorner + 10.0, grid.yllcorner, grid.crs)
    with pytest.raises(ValueError, match='cell edges'):
        T.demFromRaster(str(CA_DEM), grid=shifted)
    coarse = G.ModelGrid(37, 41, grid.cellsize * 2, grid.xllcorner, grid.yllcorner, grid.crs)
    with pytest.raises(ValueError, match='cell size'):
        T.demFromRaster(str(CA_DEM), grid=coarse)


def test_camp_branch_network_at_the_toolkit_support_area(camp_branch):
    grid, dem, terrain = camp_branch
    net = S.extractNetwork(terrain, grid, channel_threshold_km2=47571.5 / 1e6)
    assert int(net['channel_mask'].sum()) == 266
    assert len(net['segments']) == 30
    chk = S.checkTopology(net['segments'], net, terrain, grid)
    assert chk['ok'], chk['errors']
    out = chk['outlets']
    assert out['outlet_segments'] == [29]
    assert out['max_area_cell'] == (68, 68) and out['lowest_cell'] == (68, 68)
    assert out['outlet_tails'] == [(68, 68)]
    assert int(terrain['flowacc'][68, 68]) == 4334
    assert (net['segments'].loc[net['segments']['outlet'] == 0, 'save'] == True).all()


def test_camp_branch_drop_objective(camp_branch):
    grid, dem, terrain = camp_branch
    res = CI.dropAnalysis(terrain, grid, tmin_cells=10, tmax_cells=300, step_cells=10)
    assert res['objective_cells'] == 120
    assert res['band_cells'] == (120, 300)
    assert abs(res['objective_km2'] - 0.095143) < 1e-5
    tab = res['table'].set_index('cells')
    assert tab.loc[60, 'max_order'] == 3 and tab.loc[300, 'max_order'] == 2
    assert tab.loc[60, 't_abs'] > 2 and tab.loc[120, 't_abs'] < 2
