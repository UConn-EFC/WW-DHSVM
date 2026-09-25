"""Tests of streams.checkOutlets: a network that passes every format check
can still drain to a headwater; these invariants catch that.
"""
import copy

import numpy as np
import pytest

import ww_dhsvm.grid as G
import ww_dhsvm.crs as C
import ww_dhsvm.terrain as T
import ww_dhsvm.streams as S


@pytest.fixture(scope='module')
def basin():
    ny, nx, cell = 60, 80, 150.0
    grid = G.ModelGrid(ny, nx, cell, 700000.0, 4600000.0, C.from_epsg(32618))
    yy, xx = np.mgrid[0:ny, 0:nx]
    dem = (900.0 - 4.0 * yy - 2.5 * xx
           + 35 * np.sin(xx / 7.0) * np.cos(yy / 9.0)).astype('float32')
    mask = np.zeros((ny, nx), 'uint8')
    mask[2:-2, 2:-2] = 1
    grid.mask = mask
    terrain = T.conditionDEM(dem, grid)
    net = S.extractNetwork(terrain, grid, channel_threshold_km2=0.5)
    return grid, terrain, net


def _tampered(net):
    """A deep copy whose segments frame can be edited freely."""
    out = dict(net)
    out['segments'] = net['segments'].copy(deep=True)
    return out


def test_correct_network_passes(basin):
    grid, terrain, net = basin
    res = S.checkOutlets(net, terrain, grid)
    assert res['ok'], res['errors']
    assert res['max_area_segment'] in res['outlet_segments']
    assert res['lowest_segment'] in res['outlet_segments']
    assert len(res['outlet_tails']) == len(res['outlet_segments'])
    chk = S.checkTopology(net['segments'], net, terrain, grid)
    assert chk['ok'] and chk['outlets']['ok']
    # without the extra arguments the outlet check is not run
    assert S.checkTopology(net['segments'])['outlets'] is None


def test_reversed_drainage_is_caught(basin):
    """Make the outlet segment drain into a headwater: the network still
    has one outlet and no cycle, but the mouth is no longer an outlet."""
    grid, terrain, net = basin
    segs = net['segments']
    outlet_id = S.checkOutlets(net, terrain, grid)['max_area_segment']   # the mouth
    head_id = int(segs.loc[(segs['outlet'] != 0) & (segs['strahler_order'] == 1), 'ID'].iloc[0])
    bad = _tampered(net)
    b = bad['segments']
    b.loc[b['ID'] == outlet_id, 'outlet'] = head_id      # mouth drains uphill
    b.loc[b['ID'] == head_id, 'outlet'] = 0              # headwater becomes the outlet
    b.loc[b['ID'] == head_id, 'save'] = True
    b.loc[b['ID'] == outlet_id, 'save'] = False
    res = S.checkOutlets(bad, terrain, grid)
    assert not res['ok']
    assert any('basin mouth' in e for e in res['errors'])
    assert any('lowest channel cell' in e for e in res['errors'])


def test_outlet_without_save_is_caught(basin):
    grid, terrain, net = basin
    bad = _tampered(net)
    b = bad['segments']
    b.loc[b['outlet'] == 0, 'save'] = False
    res = S.checkOutlets(bad, terrain, grid)
    assert not res['ok']
    assert any('without SAVE' in e for e in res['errors'])


def test_several_outlets_give_a_warning(basin):
    grid, terrain, net = basin
    bad = _tampered(net)
    b = bad['segments']
    extra = int(b.loc[(b['outlet'] != 0), 'ID'].iloc[-1])
    b.loc[b['ID'] == extra, 'outlet'] = 0
    b.loc[b['ID'] == extra, 'save'] = True
    before = S.checkOutlets(net, terrain, grid)
    res = S.checkOutlets(bad, terrain, grid)
    assert len(res['outlet_segments']) == len(before['outlet_segments']) + 1
    assert any(f"{len(res['outlet_segments'])} outlet segments" in w for w in res['warnings'])
