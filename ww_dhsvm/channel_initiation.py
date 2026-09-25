"""Objective channel-initiation threshold by constant stream drop.

:func:`ww_dhsvm.streams.extractNetwork` places channel heads where the
upstream area first exceeds a critical support area.  That area sets the
drainage density of the whole network and, through it, how much of the
basin DHSVM treats as hillslope versus channel, so it should come from
the terrain rather than from a habit.  This module implements the
constant stream drop analysis of Broscoe (1959) and Tarboton, Bras and
Rodriguez-Iturbe (1991), as used in TauDEM and ported from the
DHSVM_Stream_Toolkit's ``drop_analysis.py``:

    The constant stream drop law holds that the mean elevation drop
    along streams of each Strahler order is statistically the same.  A
    range of support areas is swept; for each one the network is
    extracted on the D8 grid, Strahler order is assigned, and the drop
    (head elevation minus tail elevation) of every stream is measured.
    A Welch t-test compares the mean drop of first-order streams with
    that of all higher orders.  The objective support area is the start
    of the first *sustained* band of consecutive thresholds for which
    the difference is not significant (|t| below 2).  Requiring a band
    rather than the single smallest passing threshold rejects isolated
    noise dips, which are common on small basins where the t-test
    samples are small and |t| is jumpy.

The result is a physical area, A_c, so it transfers across resolutions
(cells = A_c / cell area) and, being computed on the same D8 grid the
network is built on, is consistent with :func:`extractNetwork`.

Usage::

    terrain = ww_dhsvm.terrain.conditionDEM(dem, grid)
    result = dropAnalysis(terrain, grid)          # sweep + objective
    network = ww_dhsvm.streams.extractNetwork(
        terrain, grid, channel_threshold_km2=result['objective_km2'])

or, in one step, ``extractNetwork(terrain, grid, channel_threshold_km2='drop')``.

References
----------
Broscoe, A. J. (1959). Quantitative analysis of longitudinal stream
profiles of small watersheds. Office of Naval Research, Project NR
389-042, Technical Report 18, Columbia University.

Tarboton, D. G., Bras, R. L., and Rodriguez-Iturbe, I. (1991). On the
extraction of channel networks from digital elevation data.
Hydrological Processes, 5(1), 81-100.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging
import math

import numpy as np
import pandas as pd

from ww_dhsvm.grid import ModelGrid


def welchT(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t statistic for the difference of two means (numpy only)."""
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    if a.size < 2 or b.size < 2:
        return float('nan')
    va = a.var(ddof=1) / a.size
    vb = b.var(ddof=1) / b.size
    se = math.sqrt(va + vb)
    if se == 0.0:
        return float('nan') if a.mean() == b.mean() else float('inf')
    return float((a.mean() - b.mean()) / se)


def strahlerOrder(flw, stream_mask: np.ndarray) -> np.ndarray:
    """Strahler order of the streams in ``stream_mask``, computed afresh.

    ``FlwdirRaster.stream_order`` (pyflwdir 0.5.5 through 0.5.12) caches
    the Strahler map on the object and returns the cached map on every
    later call whatever mask is passed, so a threshold sweep that calls
    it repeatedly gets the orders of the first network for all the
    others.  Calling the underlying function directly avoids the cache.
    """
    from pyflwdir import streams as _streams
    mask = np.asarray(stream_mask, dtype=bool).ravel()
    strord = _streams.strahler_order(flw.idxs_ds, flw.idxs_seq, mask=mask)
    return np.asarray(strord).reshape(flw.shape)


def streamDrops(flw, elev: np.ndarray, stream_mask: np.ndarray,
                strord: np.ndarray) -> List[Tuple[int, float]]:
    """``(strahler_order, drop)`` of every stream between junctions.

    Streams come from ``flw.streams`` (pyflwdir): one feature per reach
    from a head or a confluence down to the next confluence or the
    outlet.  The drop is the elevation at the first vertex minus the
    elevation at the last, read from ``elev`` at the cells the vertices
    fall in, exactly as the Toolkit's analysis does it.
    """
    inv = ~flw.transform
    nrows, ncols = elev.shape
    out = []
    for feat in flw.streams(mask=stream_mask, strord=strord):
        order = int(feat['properties']['strord'])
        line = feat['geometry']['coordinates']
        hx, hy = line[0]
        tx, ty = line[-1]
        hc, hr = inv * (hx, hy)
        tc, tr = inv * (tx, ty)
        hr, hc, tr, tc = int(hr), int(hc), int(tr), int(tc)
        if not (0 <= hr < nrows and 0 <= hc < ncols and
                0 <= tr < nrows and 0 <= tc < ncols):
            continue
        out.append((order, float(elev[hr, hc] - elev[tr, tc])))
    return out


def evaluateThreshold(flw, elev: np.ndarray, flowacc: np.ndarray,
                      cell_area: float, thresh_cells: int,
                      basin_cells: int) -> Optional[Dict[str, Any]]:
    """One support area: extract streams, order them, test the drops."""
    stream_mask = flowacc >= thresh_cells
    n_stream = int(stream_mask.sum())
    if n_stream < 2:
        return None
    strord = strahlerOrder(flw, stream_mask)
    rows = streamDrops(flw, elev, stream_mask, strord)
    if not rows:
        return None
    orders = np.array([r[0] for r in rows], dtype=int)
    drops = np.array([r[1] for r in rows], dtype='float64')
    o1 = drops[orders == 1]
    hi = drops[orders >= 2]
    t = welchT(o1, hi)
    t_abs = abs(t) if np.isfinite(t) else float('nan')
    cell_len = math.sqrt(cell_area)
    dd = (n_stream * cell_len) / (basin_cells * cell_area) * 1000.0
    return dict(cells=int(thresh_cells),
                area_m2=float(thresh_cells * cell_area),
                area_km2=float(thresh_cells * cell_area / 1e6),
                n_stream_cells=n_stream,
                n_streams=len(rows),
                n_order1=int(o1.size), n_higher=int(hi.size),
                max_order=int(orders.max()),
                drainage_density=float(dd),
                t_abs=float(t_abs),
                n_negative_drops=int((drops <= 0).sum()),
                passes=bool(np.isfinite(t_abs) and t_abs < 2.0))


def firstSustainedBand(results: List[Dict[str, Any]],
                       min_band: int) -> Tuple[Optional[Dict[str, Any]],
                                              List[Dict[str, Any]]]:
    """Start of the first run of at least ``min_band`` consecutive passes.

    With ``min_band=1`` this is the smallest passing threshold.
    """
    n, i = len(results), 0
    while i < n:
        if not results[i]['passes']:
            i += 1
            continue
        j = i
        while j < n and results[j]['passes']:
            j += 1
        run = results[i:j]
        if len(run) >= min_band:
            return run[0], run
        i = j
    return None, []


def dropAnalysis(terrain: Dict[str, Any],
                 grid: ModelGrid,
                 tmin_cells: int = 10,
                 tmax_cells: int = 300,
                 step_cells: int = 10,
                 min_band: int = 3,
                 use_raw_dem: bool = True) -> Dict[str, Any]:
    """Sweep support areas and pick the objective one.

    Parameters
    ----------
    terrain : dict
        Output of :func:`ww_dhsvm.terrain.conditionDEM` (``'flowdir'``,
        ``'flowacc'``, ``'dem'``, ``'raw_dem'``).
    grid : ModelGrid
        The model grid; ``grid.mask`` gives the basin area for the
        drainage density.
    tmin_cells, tmax_cells, step_cells : int, optional
        The swept thresholds, in cells.  Defaults 10 to 300 by 10.
    min_band : int, optional
        Consecutive passing thresholds that make a sustained band.
        Default 3; 1 reduces the rule to the smallest passing threshold.
    use_raw_dem : bool, optional
        Measure drops on the unconditioned DEM (default), as the Toolkit
        does; ``False`` measures them on the filled DEM.

    Returns
    -------
    dict
        ``'objective_km2'``, ``'objective_cells'`` (``None`` when no band
        passes), ``'band_cells'`` as ``(lo, hi)``, ``'passes_below_band'``
        (isolated passes rejected as noise) and ``'table'``, a DataFrame
        of the whole sweep.
    """
    flw = terrain['flowdir']
    flowacc = np.asarray(terrain['flowacc'], dtype='float64')
    elev = np.asarray(terrain['raw_dem'] if use_raw_dem and 'raw_dem' in terrain
                      else terrain['dem'], dtype='float64')
    if grid.mask is not None:
        basin_cells = int(np.count_nonzero(grid.mask))
        flowacc = np.where(grid.mask != 0, flowacc, 0.0)
    else:
        basin_cells = grid.nrows * grid.ncols
    results = []
    for thr in range(int(tmin_cells), int(tmax_cells) + 1, int(step_cells)):
        r = evaluateThreshold(flw, elev, flowacc, grid.cell_area, thr, basin_cells)
        if r is not None:
            results.append(r)
    obj, band = firstSustainedBand(results, min_band)
    below = [r['cells'] for r in results
             if r['passes'] and obj is not None and r['cells'] < obj['cells']]
    table = pd.DataFrame(results)
    out = dict(objective_km2=None if obj is None else obj['area_km2'],
               objective_cells=None if obj is None else obj['cells'],
               band_cells=None if obj is None else (band[0]['cells'], band[-1]['cells']),
               passes_below_band=below, table=table,
               tmin_cells=tmin_cells, tmax_cells=tmax_cells, step_cells=step_cells,
               min_band=min_band)
    if obj is None:
        logging.warning(f'  drop analysis: no sustained band of {min_band} consecutive '
                        f'|t| < 2 between {tmin_cells} and {tmax_cells} cells; widen '
                        f'the range or lower min_band')
    else:
        logging.info(f"  drop analysis: objective support area {obj['area_km2']:.5f} km^2 "
                     f"= {obj['cells']} cells (|t| = {obj['t_abs']:.2f}), band "
                     f"{band[0]['cells']} to {band[-1]['cells']} cells"
                     + (f', isolated passes below the band at {below} cells rejected'
                        if below else ''))
    return out


def selectSupportArea(terrain: Dict[str, Any], grid: ModelGrid, **kwargs) -> float:
    """The objective support area in km^2, or a ValueError when no band passes."""
    result = dropAnalysis(terrain, grid, **kwargs)
    if result['objective_km2'] is None:
        raise ValueError('drop analysis found no sustained band of passing '
                         'thresholds; widen tmin_cells/tmax_cells or lower '
                         'min_band, or give channel_threshold_km2 explicitly')
    return float(result['objective_km2'])
