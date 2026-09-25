"""Channel network extraction and DHSVM's three stream files.

DHSVM represents channels as a **1D network of segments superimposed on
the 2D grid**.  Three files describe it, and all three must agree
exactly or the model aborts during initialization:

``stream.class.dat``
    One row per channel *class*::

        ID   width[m]   depth[m]   manning_n   infiltration[m/s]

    Read by ``channel_read_classes`` in ``channel.c``.

``stream.network.dat``
    One row per channel *segment*, defining the routing topology::

        ID  order  slope  length[m]  class  outlet_ID  [SAVE  name]

    Read by ``channel_read_network``.  ``outlet_ID`` is the ID of the
    downstream segment, or ``0`` for the basin outlet.  ``slope`` and
    ``length`` must both be strictly positive -- ``channel.c`` treats
    zero or negative values as fatal errors.  The optional trailing
    ``SAVE`` flag and name make DHSVM write a hydrograph for that
    segment.

``stream.map.dat``
    One row per (cell, segment) intersection::

        col  row  seg_ID  length[m]  cut_height[m]  cut_width[m]  aspect[deg]  [SINK]

    Read by ``channel_grid_read_map``.  **``col`` and ``row`` are
    zero-based**, and row 0 is the northernmost row -- verified against
    the bounds check in ``channel_grid.c``, which rejects indices ``< 0``
    or ``>= channel_grid_rows/cols``.  A cell may appear on several rows
    when more than one segment crosses it; DHSVM chains these into a
    linked list.  ``cut_height`` must not exceed the cell's soil depth.

Watershed Workflow's analogous machinery (``river_tree``, ``river_mesh``,
``hydrography``) builds a vector river network for stream-aligned
meshing.  DHSVM instead needs a network that is *consistent with the
routing grid*, so ``ww_dhsvm`` derives the network from the conditioned
DEM's D8 flow directions rather than from NHD geometry.  NHD is still
used -- to burn the DEM (see :mod:`ww_dhsvm.terrain`) and to sanity-check
the result.
"""

from typing import Optional, Tuple, Dict, List, Any, Sequence
import os
import datetime
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry

from ww_dhsvm.grid import ModelGrid


# ---------------------------------------------------------------------------
# Channel classification
# ---------------------------------------------------------------------------

#: Manning's n by slope band, following PNNL's ``channelclass.py``:
#: low-gradient alluvial channels are smoother than steep boulder ones.
MANNING_BY_SLOPE = [(0.002, 0.03), (0.01, 0.05), (np.inf, 0.10)]

#: Contributing-area class breaks, m^2, from PNNL's ``channelclass.py``.
#:
#: These were designed for the Chiwawa (~500 km^2) and top out at 40 km^2.
#: Applied unchanged to a large basin every reach above 40 km^2 collapses
#: into one class -- on the 29,186 km^2 Connecticut that gave the main stem
#: a 15.5 m width where hydraulic geometry says ~230 m, a 15-fold error in
#: the channel interception area.  :func:`classifyChannels` therefore
#: derives the breaks from the basin's own area distribution by default and
#: keeps these only as an explicit opt-in.
AREA_BREAKS = [1.0e6, 1.0e7, 2.0e7, 3.0e7, 4.0e7]

#: Number of contributing-area bands when breaks are derived from the data.
N_AREA_BANDS = 6


def hydraulicGeometry(area_m2: np.ndarray,
                      width_coef: float = 0.00479,
                      width_exp: float = 0.45,
                      depth_coef: float = 0.00428,
                      depth_exp: float = 0.30,
                      min_width: float = 0.5,
                      min_depth: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Channel width and depth from contributing area.

    Downstream hydraulic geometry (Leopold & Maddock, 1953) gives channel
    dimensions as power laws in discharge; substituting the standard
    regional relation between bankfull discharge and drainage area yields
    power laws in area directly::

        W = a A^b      D = c A^d

    with ``A`` in m^2.  The defaults are the humid-temperate / New
    England form ``W = 2.4 A_km2^0.45`` and ``D = 0.27 A_km2^0.30``,
    rewritten for area in m^2.  They give:

    ======================  ==========  ==========
    drainage area           width       depth
    ======================  ==========  ==========
    1 km^2                  2.4 m       0.27 m
    100 km^2                19 m        1.1 m
    1 000 km^2              54 m        2.1 m
    25 000 km^2 (whole CT)  229 m       5.6 m
    ======================  ==========  ==========

    These are *regional* relations and should be recalibrated wherever
    bankfull surveys or HYDRoSWOT observations exist; they set the
    channel's interception area and its routing celerity, so they matter
    for peak-flow timing.

    Parameters
    ----------
    area_m2 : np.ndarray
        Contributing area, m^2.
    width_coef, width_exp : float, optional
        Coefficient and exponent of the width relation.
    depth_coef, depth_exp : float, optional
        Coefficient and exponent of the depth relation.
    min_width, min_depth : float, optional
        Floors, metres.  DHSVM rejects non-positive channel dimensions.

    Returns
    -------
    width, depth : np.ndarray
        Metres.
    """
    a = np.maximum(np.asarray(area_m2, dtype='float64'), 1.0)
    width = np.maximum(width_coef * np.power(a, width_exp), min_width)
    depth = np.maximum(depth_coef * np.power(a, depth_exp), min_depth)
    return width, depth


def classifyChannels(slope: np.ndarray,
                     mean_area: np.ndarray,
                     area_breaks: Optional[Sequence[float]] = None,
                     n_bands: int = N_AREA_BANDS,
                     **hg_kwargs) -> Tuple[np.ndarray, pd.DataFrame]:
    """Assign each segment a channel class and build the class table.

    Keeps the structure of PNNL's ``channelclass.py`` -- 3 slope bands x
    ``n_bands`` contributing-area bands -- but derives the area breaks
    from **this basin's own distribution** rather than using the fixed
    ones tuned for the Chiwawa.

    That matters as soon as a basin is large.  PNNL's breaks stop at
    40 km^2; on the 29,186 km^2 Connecticut every reach from a small
    tributary to the main stem then falls in one class and inherits a
    15.5 m width, where downstream hydraulic geometry gives ~230 m at the
    mouth.  Since DHSVM multiplies channel length by width to get the
    interception area, that is a fifteen-fold error in how much water the
    main stem can receive.

    Breaks are placed at geometric (log-spaced) quantiles of the segment
    contributing areas, which suits a quantity spanning several orders of
    magnitude, and each class takes the geometric-mean area of its band.

    Parameters
    ----------
    slope : np.ndarray
        Segment slope, dimensionless (m/m).
    mean_area : np.ndarray
        Segment mean contributing area, m^2.
    area_breaks : sequence of float, optional
        Explicit break points, m^2.  Pass :data:`AREA_BREAKS` to
        reproduce the original PNNL scheme exactly.
    n_bands : int, optional
        Number of area bands when breaks are derived.  Default 6, giving
        the familiar 18 classes.
    **hg_kwargs
        Forwarded to :func:`hydraulicGeometry`.

    Returns
    -------
    classes : np.ndarray
        Class ID per segment, ``1 .. 3*n_bands``.
    class_table : pd.DataFrame
        Columns ``ID``, ``width``, ``depth``, ``manning``,
        ``infiltration``, ready for :func:`writeStreamClassFile`.
    """
    slope = np.asarray(slope, dtype='float64')
    area = np.asarray(mean_area, dtype='float64')

    if area_breaks is None:
        pos = area[np.isfinite(area) & (area > 0)]
        if pos.size < n_bands:
            area_breaks = list(AREA_BREAKS)
            n_bands = 6
        else:
            # Log-spaced, not quantile-spaced.  Contributing area is
            # power-law distributed -- the vast majority of reaches are
            # small headwaters -- so quantile breaks pile five of six
            # boundaries below 35 km^2 and leave one band spanning
            # 35 km^2 to the basin outlet.  Width goes as A^0.45, so
            # equal-width bands in log(A) keep the width error bounded
            # across the whole network instead of concentrating it on the
            # main stem, which is where it matters most.
            lo, hi = np.percentile(pos, 1), pos.max()
            area_breaks = list(np.geomspace(lo, hi, n_bands + 1)[1:-1])
            n_bands = len(area_breaks) + 1
        logging.info('  channel-area class breaks (km^2): '
                     + ', '.join(f'{b/1e6:.3g}' for b in area_breaks))
    else:
        area_breaks = list(area_breaks)
        n_bands = len(area_breaks) + 1

    slope_band = np.digitize(slope, [b for b, _ in MANNING_BY_SLOPE[:-1]])
    area_band = np.digitize(area, area_breaks)

    classes = slope_band * n_bands + area_band + 1

    # Representative area per band: geometric midpoint, with the open
    # ends anchored on the observed minimum and maximum so the largest
    # class really does describe the main stem.
    pos = area[np.isfinite(area) & (area > 0)]
    lo = float(pos.min()) if pos.size else 1.0e5
    hi = float(pos.max()) if pos.size else 1.0e8
    edges = [lo] + area_breaks + [hi]
    reps = [np.sqrt(edges[i] * edges[i + 1]) for i in range(n_bands)]

    rows = []
    for sb, (_, n) in enumerate(MANNING_BY_SLOPE):
        for ab in range(n_bands):
            cid = sb * n_bands + ab + 1
            w, d = hydraulicGeometry(np.array([reps[ab]]), **hg_kwargs)
            rows.append(dict(ID=cid, width=float(w[0]), depth=float(d[0]),
                             manning=n, infiltration=0.0))
    class_table = pd.DataFrame(rows).sort_values('ID').reset_index(drop=True)

    logging.info(f'  channel classes: {len(class_table)} defined, '
                 f'{len(np.unique(classes))} used; '
                 f'width {class_table.width.min():.2f}-{class_table.width.max():.2f} m')
    return classes, class_table


# ---------------------------------------------------------------------------
# Network extraction
# ---------------------------------------------------------------------------

def extractNetwork(terrain: Dict[str, Any],
                   grid: ModelGrid,
                   channel_threshold_km2: float | str = 1.0,
                   min_segment_cells: int = 2,
                   min_slope: float = 1.0e-4,
                   gauge_rowcol: Optional[Tuple[int, int]] = None,
                   drop_kwargs: Optional[Dict[str, Any]] = None,
                   **hg_kwargs) -> Dict[str, Any]:
    """Delineate a channel network and build all DHSVM stream tables.

    The channel head is placed where upstream area first exceeds
    ``channel_threshold_km2``.  This "constant critical support area" rule
    is the standard approach and is what the DHSVM AML toolchain used; it
    is a genuine modelling choice, since it sets how much of the basin is
    treated as hillslope versus channel, and it should be checked against
    mapped drainage density (see :func:`compareToReference`).  Pass
    ``channel_threshold_km2='drop'`` to let the constant stream drop
    analysis choose the area from the terrain
    (:func:`ww_dhsvm.channel_initiation.dropAnalysis`; ``drop_kwargs``
    are passed on to it, and the sweep is returned under
    ``'drop_analysis'``).

    Parameters
    ----------
    terrain : dict
        Output of :func:`ww_dhsvm.terrain.conditionDEM`.
    grid : ModelGrid
        The model grid.  ``grid.mask`` limits the network to the basin.
    channel_threshold_km2 : float, optional
        Critical support area for channel initiation, km^2.  Default 1.0.
    min_segment_cells : int, optional
        Segments shorter than this are merged downstream, which avoids
        degenerate one-cell segments whose slope is ill-determined.
    min_slope : float, optional
        Floor on channel slope, m/m.  Default ``1e-4`` (0.1 m/km), about
        the gradient of the Connecticut River near its mouth and a
        reasonable lower bound for a real alluvial channel.  **This is
        not a cosmetic clamp.**  Depression filling flattens valley
        floors to exactly constant elevation, so reaches there compute a
        drop of zero; left at a token 1e-5 they route water at
        essentially zero Manning velocity, and the basin's own outflow
        stalls.  Reaches below the floor first have their slope
        re-estimated from the downstream valley gradient (see
        :func:`_refineFlatSlopes`), and only then is the floor applied.
    gauge_rowcol : tuple of int, optional
        ``(row, col)`` of a streamflow gauge.  The segment containing it
        is flagged ``SAVE`` so DHSVM writes its hydrograph.

    Returns
    -------
    dict
        ``'segments'`` (DataFrame), ``'class_table'`` (DataFrame),
        ``'map'`` (DataFrame), ``'channel_mask'``, ``'segment_id_grid'``,
        ``'cut_height_grid'``, ``'cut_width_grid'``.
    """
    flw = terrain['flowdir']
    uparea = terrain['uparea']
    dem = terrain['dem']
    mask = grid.mask if grid.mask is not None else np.ones(grid.shape, dtype='uint8')

    drop = None
    if isinstance(channel_threshold_km2, str):
        if channel_threshold_km2.lower() != 'drop':
            raise ValueError("channel_threshold_km2 must be a number in km^2 or 'drop'")
        from ww_dhsvm.channel_initiation import dropAnalysis
        drop = dropAnalysis(terrain, grid, **(drop_kwargs or {}))
        if drop['objective_km2'] is None:
            raise ValueError('the drop analysis found no sustained band of passing '
                             'thresholds; widen its range with drop_kwargs or give '
                             'channel_threshold_km2 explicitly')
        channel_threshold_km2 = float(drop['objective_km2'])
        logging.info(f'  channel threshold from the drop analysis: '
                     f'{channel_threshold_km2:.5f} km^2 ({drop["objective_cells"]} cells)')

    thresh_m2 = channel_threshold_km2 * 1e6
    stream_mask = (uparea >= thresh_m2) & (mask != 0)
    n_channel = int(stream_mask.sum())
    if n_channel == 0:
        raise ValueError(
            f'No cells exceed the {channel_threshold_km2} km^2 channel-initiation '
            f'threshold.  Max upstream area is {np.nanmax(uparea)/1e6:.2f} km^2 -- '
            f'lower the threshold.')

    drainage_density = n_channel * grid.cellsize / (grid.basin_area_km2 * 1e6) * 1e3
    logging.info(f'  channel cells: {n_channel} '
                 f'({100.0*n_channel/max(int(mask.sum()),1):.1f}% of basin), '
                 f'drainage density {drainage_density:.3f} km/km^2')

    # --- split the channel mask into segments at confluences ---------
    seg_id_grid, segments = _buildSegments(flw, stream_mask, uparea, dem, grid,
                                           min_segment_cells)
    logging.info(f'  built {len(segments)} channel segments')

    # Reaches on depression-filled flats have zero computed drop; recover a
    # physical gradient for them before anything downstream uses slope.
    segments = _refineFlatSlopes(segments, min_slope)

    # --- classify -----------------------------------------------------
    classes, class_table = classifyChannels(segments['slope'].values,
                                            segments['mean_area'].values,
                                            **hg_kwargs)
    segments['class'] = classes
    cls_lookup = class_table.set_index('ID')
    segments['width'] = cls_lookup.loc[segments['class'], 'width'].values
    segments['depth'] = cls_lookup.loc[segments['class'], 'depth'].values

    # --- flag the gauge segment --------------------------------------
    segments['save'] = False
    segments['save_name'] = ''
    if gauge_rowcol is not None:
        gr, gc = gauge_rowcol
        if 0 <= gr < grid.nrows and 0 <= gc < grid.ncols and seg_id_grid[gr, gc] > 0:
            sid = int(seg_id_grid[gr, gc])
            segments.loc[segments['ID'] == sid, 'save'] = True
            segments.loc[segments['ID'] == sid, 'save_name'] = 'GAUGE'
            logging.info(f'  gauge at ({gr},{gc}) lies on segment {sid}; flagged SAVE')
        else:
            logging.warning(f'  gauge at ({gr},{gc}) is not on a channel cell; '
                            f'no SAVE flag written')

    # Always save the outlet segment so there is at least one hydrograph.
    outlet_ids = segments.loc[segments['outlet'] == 0, 'ID'].tolist()
    for oid in outlet_ids:
        segments.loc[segments['ID'] == oid, 'save'] = True
        if not segments.loc[segments['ID'] == oid, 'save_name'].values[0]:
            segments.loc[segments['ID'] == oid, 'save_name'] = f'OUTLET_{oid}'
    logging.info(f'  {len(outlet_ids)} outlet segment(s): {outlet_ids}')

    # --- the cell-by-cell stream map ---------------------------------
    stream_map, cut_h, cut_w = _buildStreamMap(seg_id_grid, segments, terrain, grid)
    logging.info(f'  stream map: {len(stream_map)} (cell, segment) records')

    return dict(segments=segments, class_table=class_table, map=stream_map,
                channel_mask=stream_mask, segment_id_grid=seg_id_grid,
                cut_height_grid=cut_h, cut_width_grid=cut_w,
                drainage_density=drainage_density,
                channel_threshold_km2=channel_threshold_km2,
                drop_analysis=drop)


def _buildSegments(flw, stream_mask, uparea, dem, grid, min_segment_cells):
    """Split the channel mask into topologically linked segments."""
    nrows, ncols = grid.shape
    n = nrows * ncols

    # pyflwdir gives, for each cell, the flat index of its downstream cell.
    idxs_ds = flw.idxs_ds.reshape(nrows, ncols)

    flat_stream = stream_mask.ravel()
    stream_idx = np.flatnonzero(flat_stream)

    # Count channel inflows per cell to find confluences.
    inflow = np.zeros(n, dtype='int32')
    ds_flat = idxs_ds.ravel()
    for i in stream_idx:
        d = ds_flat[i]
        if d != i and 0 <= d < n and flat_stream[d]:
            inflow[d] += 1

    # A new segment starts at a source (no channel inflow) or just below a
    # confluence (>=2 channel inflows).
    is_start = np.zeros(n, dtype=bool)
    is_start[stream_idx] = (inflow[stream_idx] == 0) | (inflow[stream_idx] >= 2)

    seg_id_flat = np.zeros(n, dtype='int32')
    seg_cells: Dict[int, List[int]] = {}
    next_id = 1

    for start in np.flatnonzero(is_start):
        if not flat_stream[start]:
            continue
        cells = []
        cur = start
        while True:
            if seg_id_flat[cur] != 0:
                break
            cells.append(cur)
            seg_id_flat[cur] = next_id
            nxt = ds_flat[cur]
            if nxt == cur or not (0 <= nxt < n) or not flat_stream[nxt]:
                break                      # left the basin / reached outlet
            if is_start[nxt]:
                break                      # next cell begins a new segment
            cur = nxt
        if cells:
            seg_cells[next_id] = cells
            next_id += 1

    # Any channel cell not yet assigned (rare: cycles broken by filling)
    # gets folded into a trailing segment so the map stays complete.
    leftover = [i for i in stream_idx if seg_id_flat[i] == 0]
    if leftover:
        logging.warning(f'  {len(leftover)} channel cells were unreachable from any '
                        f'source; assigning them to singleton segments')
        for i in leftover:
            seg_id_flat[i] = next_id
            seg_cells[next_id] = [i]
            next_id += 1

    # --- per-segment geometry and topology ---------------------------
    diag = np.sqrt(2.0) * grid.cellsize
    rows = []
    for sid, cells in seg_cells.items():
        z = dem.ravel()[cells]
        # Length: sum of cell-to-cell steps, cardinal vs diagonal.
        length = 0.0
        for k, c in enumerate(cells):
            d = ds_flat[c]
            if d == c or not (0 <= d < n):
                length += grid.cellsize
                continue
            dr = abs(d // ncols - c // ncols)
            dc = abs(d % ncols - c % ncols)
            length += diag if (dr and dc) else grid.cellsize

        drop = float(z[0] - z[-1])
        # Raw slope; may be zero on a depression-filled flat.  The floor
        # and the flat-reach re-estimate are applied by _refineFlatSlopes.
        slope = drop / max(length, 1e-6)

        last = cells[-1]
        ds_cell = ds_flat[last]
        if ds_cell == last or not (0 <= ds_cell < n) or not flat_stream[ds_cell]:
            outlet = 0
        else:
            outlet = int(seg_id_flat[ds_cell])
            if outlet == sid:
                outlet = 0

        rows.append(dict(
            ID=int(sid), outlet=outlet, n_cells=len(cells),
            length=float(length), slope=float(slope), drop=drop,
            mean_area=float(np.mean(uparea.ravel()[cells])),
            max_area=float(np.max(uparea.ravel()[cells])),
            head_cell=int(cells[0]), tail_cell=int(last)))

    segments = pd.DataFrame(rows).sort_values('ID').reset_index(drop=True)

    # Two different orderings, for two different jobs:
    #  - strahler_order  : hydrologic stream order, for classification and plots
    #  - order           : DHSVM's routing rank, written to stream.network.dat
    segments['strahler_order'] = _strahlerOrder(segments)
    segments['order'] = _routingOrder(segments)

    seg_id_grid = seg_id_flat.reshape(nrows, ncols)
    return seg_id_grid, segments



def _refineFlatSlopes(segments: pd.DataFrame, min_slope: float) -> pd.DataFrame:
    """Give depression-filled flat reaches a physical slope.

    Filling a depression raises every cell in it to the spill elevation,
    so a reach lying inside one has *exactly* zero drop -- not a small
    drop, zero.  Along a low-relief valley floor such as the Connecticut's
    that is a quarter of the network -- 5,923 of 23,837 reaches (24.8%)
    on the full basin -- and a channel with no gradient transports
    nothing: water routed into it accumulates instead of reaching the
    outlet.

    Rather than clamp such reaches to an arbitrary constant, this walks
    **downstream** from each flat reach, accumulating length and elevation
    drop until the accumulated gradient reaches ``min_slope`` or the
    outlet is reached, and assigns that gradient.  The result is the local
    *valley* slope, which is the physically meaningful quantity: a reach
    on a filled flat really does sit in a valley that descends, the DEM
    just fails to resolve it over one reach length.

    Reaches for which even the whole downstream path yields less than
    ``min_slope`` -- the lowest reaches of a very flat basin -- take the
    floor.

    Parameters
    ----------
    segments : pd.DataFrame
        Segment table with ``ID``, ``outlet``, ``slope``, ``length`` and
        ``drop``.  Modified and returned.
    min_slope : float
        Target and floor gradient, m/m.

    Returns
    -------
    pd.DataFrame
        ``segments`` with ``slope`` repaired.
    """
    n_flat = int((segments['slope'] < min_slope).sum())
    if n_flat == 0:
        return segments

    length_by_id = dict(zip(segments['ID'], segments['length']))
    drop_by_id = dict(zip(segments['ID'], segments['drop']))
    outlet_by_id = dict(zip(segments['ID'].astype(int),
                            segments['outlet'].astype(int)))

    new_slope = segments['slope'].to_numpy(dtype='float64').copy()
    n_from_valley, n_floored = 0, 0

    for i, (sid, sl) in enumerate(zip(segments['ID'].astype(int), new_slope)):
        if sl >= min_slope:
            continue
        acc_len = float(length_by_id[sid])
        acc_drop = max(float(drop_by_id[sid]), 0.0)
        cur, steps = outlet_by_id.get(sid, 0), 0
        while cur != 0 and steps < 200:
            acc_len += float(length_by_id.get(cur, 0.0))
            acc_drop += max(float(drop_by_id.get(cur, 0.0)), 0.0)
            if acc_len > 0 and acc_drop / acc_len >= min_slope:
                break
            cur = outlet_by_id.get(cur, 0)
            steps += 1
        est = acc_drop / acc_len if acc_len > 0 else 0.0
        if est >= min_slope:
            new_slope[i] = est
            n_from_valley += 1
        else:
            new_slope[i] = min_slope
            n_floored += 1

    segments = segments.copy()
    segments['slope'] = new_slope
    logging.info(f'  {n_flat:,} reaches ({100.0*n_flat/len(segments):.1f}%) had a '
                 f'slope below {min_slope:g} after depression filling: '
                 f'{n_from_valley:,} re-estimated from the downstream valley '
                 f'gradient, {n_floored:,} set to the floor')
    return segments



def _routingOrder(segments: pd.DataFrame) -> np.ndarray:
    """Compute DHSVM's channel **routing rank**.

    This is *not* Strahler order, and the difference is not cosmetic --
    it decides whether the channel network conserves mass.

    ``channel_route_network`` in ``channel.c`` sweeps the segment list by
    ascending ``order``::

        for (order = 1; ; order += 1) {
            for every segment with current->order == order:
                channel_route_segment(...)      # pushes outflow into
                                                # segment->outlet->inflow
            if (no segment had this order) break;
        }

    A segment is therefore routed at its own rank, and its outflow is
    only *then* added to its downstream neighbour's inflow.  For that to
    work, every segment must be routed **strictly after** all of its
    upstream contributors -- so ``order`` has to satisfy

        order(s) = 1 + max(order(u) for u upstream of s)

    with headwaters at 1.  Strahler order does not satisfy it: two
    order-2 reaches merging give an order-3 reach, but an order-1 reach
    joining an order-3 leaves it order-3, so contributor and receiver
    share a rank.  The receiver is then routed in the same sweep as its
    contributor -- often before it -- and since
    ``channel_step_initialize_network`` zeroes ``inflow`` each timestep,
    the late contribution is silently dropped.

    Measured on the Connecticut River Basin, using Strahler order lost
    **96% of the water entering the channel network**: 86 mm/yr of
    lateral inflow produced 1.6 mm/yr of outflow, the remainder absorbed
    by DHSVM's channel closure error, while the cell-level mass balance
    stayed perfect at 2e-4 mm throughout.  The full basin's routing rank
    reaches **673** across 23,837 segments.  DHSVM's own Chiwawa test case confirms the convention -- its
    ``order`` column is ``1 + max(upstream)`` for all 494 segments and
    reaches 69, far beyond any Strahler value.

    The ranks are contiguous by construction (a segment of rank *k* > 1
    has an upstream of rank *k*-1), which matters because the sweep stops
    at the first empty rank.

    Parameters
    ----------
    segments : pd.DataFrame
        Needs ``ID`` and ``outlet``.

    Returns
    -------
    np.ndarray
        Routing rank per segment, ``>= 1``.
    """
    ids = segments['ID'].to_numpy(dtype='int64')
    outlets = segments['outlet'].to_numpy(dtype='int64')
    idset = set(int(i) for i in ids)

    upstream = defaultdict(list)
    n_up = defaultdict(int)
    for sid, out in zip(ids, outlets):
        if out != 0 and int(out) in idset:
            upstream[int(out)].append(int(sid))
            n_up[int(out)] += 1

    # Kahn's algorithm from the headwaters downstream.
    rank = {int(i): 1 for i in ids}
    remaining = {int(i): n_up.get(int(i), 0) for i in ids}
    queue = [int(i) for i in ids if remaining[int(i)] == 0]
    outlet_of = dict(zip((int(i) for i in ids), (int(o) for o in outlets)))

    seen = 0
    while queue:
        sid = queue.pop()
        seen += 1
        down = outlet_of.get(sid, 0)
        if down == 0 or down not in remaining:
            continue
        rank[down] = max(rank[down], rank[sid] + 1)
        remaining[down] -= 1
        if remaining[down] == 0:
            queue.append(down)

    if seen != len(ids):
        # Only reachable if a cycle survived checkTopology; rank what we can.
        logging.warning(f'  routing rank: {len(ids) - seen} segments lie on a '
                        f'cycle and keep rank 1; fix the topology first')

    out = np.array([rank[int(i)] for i in ids], dtype='int32')
    logging.info(f'  routing rank: 1 to {out.max()} over {len(out):,} segments '
                 f'(DHSVM routes by ascending rank, so this must be '
                 f'1 + max(upstream), not Strahler order)')
    return out


def _strahlerOrder(segments: pd.DataFrame) -> np.ndarray:
    """Strahler stream order over the segment topology."""
    children = defaultdict(list)
    for sid, out in zip(segments['ID'], segments['outlet']):
        if out != 0:
            children[out].append(sid)

    order = {}

    def _order(sid, depth=0):
        if sid in order:
            return order[sid]
        if depth > 100000:
            order[sid] = 1
            return 1
        kids = children.get(sid, [])
        if not kids:
            order[sid] = 1
        else:
            kid_orders = sorted((_order(k, depth + 1) for k in kids), reverse=True)
            if len(kid_orders) >= 2 and kid_orders[0] == kid_orders[1]:
                order[sid] = kid_orders[0] + 1
            else:
                order[sid] = kid_orders[0]
        return order[sid]

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 200000))
    try:
        return np.array([_order(s) for s in segments['ID']], dtype='int32')
    finally:
        sys.setrecursionlimit(old_limit)


def _buildStreamMap(seg_id_grid, segments, terrain, grid):
    """Build the per-cell stream map plus cut-height/width grids.

    Written with lookup arrays rather than per-cell DataFrame indexing: a
    large basin has tens of thousands of channel cells, and a pandas
    ``.loc`` per cell turns a one-second operation into a minute.
    """
    aspect = terrain['aspect']

    max_id = int(segments['ID'].max()) + 1
    len_by_id = np.zeros(max_id, dtype='float64')
    depth_by_id = np.zeros(max_id, dtype='float64')
    width_by_id = np.zeros(max_id, dtype='float64')
    ids = segments['ID'].to_numpy(dtype='int64')
    # Per-cell channel length: the segment's length shared out over its
    # cells.  This conserves total channel length exactly, which matters
    # because DHSVM multiplies length by width to get the channel's
    # interception area.
    len_by_id[ids] = (segments['length'].to_numpy()
                      / np.maximum(segments['n_cells'].to_numpy(), 1))
    depth_by_id[ids] = segments['depth'].to_numpy()
    width_by_id[ids] = segments['width'].to_numpy()

    rr, cc = np.nonzero(seg_id_grid > 0)
    sids = seg_id_grid[rr, cc].astype('int64')
    valid = sids < max_id
    rr, cc, sids = rr[valid], cc[valid], sids[valid]

    stream_map = pd.DataFrame({
        'col': cc.astype('int32'),
        'row': rr.astype('int32'),
        'seg_id': sids.astype('int32'),
        'length': len_by_id[sids],
        'cut_height': depth_by_id[sids],
        'cut_width': width_by_id[sids],
        'aspect': aspect[rr, cc],
    })

    stream_map = _capChannelAreaToCell(stream_map, grid)

    cut_h = np.zeros(grid.shape, dtype='float64')
    cut_w = np.zeros(grid.shape, dtype='float64')
    if len(stream_map):
        cut_h[stream_map['row'].to_numpy(), stream_map['col'].to_numpy()] = \
            stream_map['cut_height'].to_numpy()
        cut_w[stream_map['row'].to_numpy(), stream_map['col'].to_numpy()] = \
            stream_map['cut_width'].to_numpy()
        stream_map = stream_map.sort_values(['row', 'col']).reset_index(drop=True)
    return stream_map, cut_h, cut_w


def _capChannelAreaToCell(stream_map: pd.DataFrame, grid: ModelGrid,
                          max_fraction: float = 0.9) -> pd.DataFrame:
    """Stop a channel from occupying more than the cell that contains it.

    ``ChannelCut`` in ``DHSVMChannel.c`` sets a cell's channel area to
    ``channel_grid_cell_width * channel_grid_cell_length`` -- the
    length-weighted mean width times the total channel length in the cell
    -- and **never clamps it to the cell area**.  A channel wider than the
    grid spacing therefore intercepts precipitation over more ground than
    the cell actually has.

    That is not a hypothetical.  On the Connecticut at 150 m, downstream
    hydraulic geometry gives the main stem ~230 m of width, so each
    main-stem cell claimed ~117% of its own area.  The resulting water
    balance was impossible: channel interception of 1,153 mm against
    955 mm of precipitation, and a runoff ratio of 1.19.

    The real constraint is a resolution limit -- **a grid cannot
    represent a river wider than its cells** -- so the honest response is
    to cap the width and say so, rather than to let the model invent
    water.  Widths in an over-subscribed cell are scaled down in
    proportion so the channel occupies at most ``max_fraction`` of it,
    leaving a little land in the cell to generate the runoff that feeds
    the channel.

    When this binds on many cells the message is that the grid is too
    coarse for the river being modelled; the fix is a finer grid, not a
    bigger cap.

    Parameters
    ----------
    stream_map : pd.DataFrame
        Per-cell records with ``row``, ``col``, ``length``, ``cut_width``.
    grid : ModelGrid
        Supplies the cell area.
    max_fraction : float, optional
        Largest share of a cell the channel may occupy.  Default 0.9.

    Returns
    -------
    pd.DataFrame
        ``stream_map`` with ``cut_width`` capped where necessary.
    """
    if not len(stream_map):
        return stream_map

    limit = max_fraction * grid.cell_area
    key = (stream_map['row'].to_numpy().astype('int64') * grid.ncols
           + stream_map['col'].to_numpy().astype('int64'))
    area = stream_map['cut_width'].to_numpy() * stream_map['length'].to_numpy()

    order = np.argsort(key, kind='stable')
    k_sorted, a_sorted = key[order], area[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    totals = np.add.reduceat(a_sorted, starts)
    counts = np.diff(np.r_[starts, len(k_sorted)])
    cell_total = np.repeat(totals, counts)

    scale_sorted = np.where(cell_total > limit, limit / cell_total, 1.0)
    scale = np.empty_like(scale_sorted)
    scale[order] = scale_sorted

    n_capped = int((scale < 1.0).sum())
    if n_capped:
        widest = float(stream_map['cut_width'].max())
        eff_max = limit / float(np.median(stream_map['length']))
        stream_map = stream_map.copy()
        stream_map['cut_width'] = stream_map['cut_width'].to_numpy() * scale
        logging.warning(
            f'  channel width capped in {n_capped:,} of {len(stream_map):,} cells: '
            f'the widest class ({widest:.0f} m) exceeds what a {grid.cellsize:g} m '
            f'grid can hold (~{eff_max:.0f} m).  DHSVM does not clamp channel area '
            f'to cell area, so uncapped this would intercept more water than falls '
            f'on the cell.  A finer grid is the real fix.')
    return stream_map


# ---------------------------------------------------------------------------
# Topology validation
# ---------------------------------------------------------------------------

def checkOutlets(network: Dict[str, Any],
                 terrain: Dict[str, Any],
                 grid: ModelGrid) -> Dict[str, Any]:
    """Check that the network drains to the basin mouth.

    Three invariants that a network can violate while still passing every
    format check (a reversed or mis-linked network routes water to a
    headwater and DHSVM runs without complaint):

    - the cell with the largest upstream area lies in an outlet segment
      (``outlet == 0``);
    - the lowest channel cell lies in an outlet segment;
    - every outlet segment is flagged ``SAVE``, so its hydrograph is
      written.

    Also reported, as information: the outlet tail cells and how many of
    them sit on the basin-mask boundary.  Several outlets are legitimate
    when the mask really spans several drainages, and a warning when it
    should not.

    Parameters
    ----------
    network : dict
        Output of :func:`extractNetwork`.
    terrain : dict
        Output of :func:`ww_dhsvm.terrain.conditionDEM`.
    grid : ModelGrid

    Returns
    -------
    dict
        ``'ok'``, ``'errors'``, ``'warnings'``, ``'max_area_cell'``,
        ``'max_area_segment'``, ``'lowest_cell'``, ``'lowest_segment'``,
        ``'outlet_segments'``, ``'outlet_tails'``, ``'n_edge_outlets'``.
    """
    errors, warnings = [], []
    segs = network['segments']
    sid = np.asarray(network['segment_id_grid'])
    mask = (grid.mask != 0) if grid.mask is not None else np.ones(grid.shape, bool)
    uparea = np.where(mask, np.asarray(terrain['uparea'], dtype='float64'), -np.inf)
    dem = np.asarray(terrain['dem'], dtype='float64')
    nrows, ncols = grid.shape

    outlet_ids = set(int(i) for i in segs.loc[segs['outlet'] == 0, 'ID'])
    r, c = np.unravel_index(int(np.argmax(uparea)), uparea.shape)
    max_seg = int(sid[r, c])
    if max_seg == 0:
        errors.append(f'the largest-area cell ({r},{c}) is not a channel cell')
    elif max_seg not in outlet_ids:
        errors.append(f'the largest-area cell ({r},{c}) lies in segment {max_seg}, '
                      f'which drains to segment '
                      f"{int(segs.loc[segs['ID'] == max_seg, 'outlet'].values[0])} "
                      f'instead of being an outlet: the network does not drain to '
                      f'the basin mouth')

    on_channel = sid > 0
    if on_channel.any():
        z = np.where(on_channel, dem, np.inf)
        lr, lc = np.unravel_index(int(np.argmin(z)), z.shape)
        low_seg = int(sid[lr, lc])
        if low_seg not in outlet_ids:
            errors.append(f'the lowest channel cell ({lr},{lc}, {dem[lr, lc]:.2f} m) '
                          f'lies in segment {low_seg}, which is not an outlet')
    else:
        lr, lc, low_seg = -1, -1, 0
        errors.append('no channel cells')

    if 'save' in segs.columns:
        unsaved = sorted(int(i) for i in segs.loc[(segs['outlet'] == 0) & ~segs['save'].astype(bool), 'ID'])
        if unsaved:
            errors.append(f'outlet segments without SAVE: {unsaved[:10]}')

    tails, n_edge = [], 0
    if 'tail_cell' in segs.columns:
        for _, s in segs[segs['outlet'] == 0].iterrows():
            tr, tc = divmod(int(s['tail_cell']), ncols)
            tails.append((tr, tc))
            r0, r1 = max(tr - 1, 0), min(tr + 2, nrows)
            c0, c1 = max(tc - 1, 0), min(tc + 2, ncols)
            if (not mask[r0:r1, c0:c1].all()) or tr in (0, nrows - 1) or tc in (0, ncols - 1):
                n_edge += 1
    if len(outlet_ids) > 1:
        warnings.append(f'{len(outlet_ids)} outlet segments, {n_edge} of them ending on '
                        f'the mask boundary; expected for several drainages, a sign of a '
                        f'mask past the divide for one watershed')

    result = dict(ok=(len(errors) == 0), errors=errors, warnings=warnings,
                  max_area_cell=(int(r), int(c)), max_area_segment=max_seg,
                  lowest_cell=(int(lr), int(lc)), lowest_segment=low_seg,
                  outlet_segments=sorted(outlet_ids), outlet_tails=tails,
                  n_edge_outlets=n_edge)
    if result['ok']:
        logging.info(f'  outlets OK: largest-area cell {result["max_area_cell"]} and '
                     f'lowest channel cell {result["lowest_cell"]} lie in outlet '
                     f'segment(s) {result["outlet_segments"]}')
    else:
        for e in errors:
            logging.error(f'  OUTLET ERROR: {e}')
    for w in warnings:
        logging.warning(f'  outlet warning: {w}')
    return result


def checkTopology(segments: pd.DataFrame,
                  network: Optional[Dict[str, Any]] = None,
                  terrain: Optional[Dict[str, Any]] = None,
                  grid: Optional[ModelGrid] = None) -> Dict[str, Any]:
    """Validate the segment network against DHSVM's requirements.

    ``channel.c`` aborts on duplicate IDs, on an ``outlet`` that names no
    existing segment, and on non-positive slope or length.  A cycle would
    make ``channel_route_network`` recurse forever.  Catching these here,
    with an actionable message, is far cheaper than debugging a DHSVM
    abort.

    When ``network``, ``terrain`` and ``grid`` are all given, the outlet
    invariants of :func:`checkOutlets` are checked too and their errors
    count against ``'ok'``.

    Returns
    -------
    dict
        ``'ok'`` plus a list of problems under ``'errors'`` and
        ``'warnings'``, and summary counts.
    """
    errors, warnings = [], []
    ids = segments['ID'].values

    dup = pd.Series(ids).duplicated()
    if dup.any():
        errors.append(f'{int(dup.sum())} duplicate segment IDs: '
                      f'{sorted(set(ids[dup.values]))[:10]}')

    idset = set(int(i) for i in ids)
    bad_out = [int(o) for o in segments['outlet'] if o != 0 and int(o) not in idset]
    if bad_out:
        errors.append(f'{len(bad_out)} segments name a non-existent outlet: '
                      f'{sorted(set(bad_out))[:10]}')

    if (segments['slope'] <= 0).any():
        errors.append(f"{int((segments['slope'] <= 0).sum())} segments have "
                      f'slope <= 0 (DHSVM requires strictly positive)')
    if (segments['length'] <= 0).any():
        errors.append(f"{int((segments['length'] <= 0).sum())} segments have "
                      f'length <= 0 (DHSVM requires strictly positive)')

    # Cycle detection by walking downstream from every segment.
    outlet_of = dict(zip(segments['ID'].astype(int), segments['outlet'].astype(int)))
    cycles = []
    for sid in idset:
        seen, cur, steps = set(), sid, 0
        while cur != 0 and steps < len(idset) + 5:
            if cur in seen:
                cycles.append(sid)
                break
            seen.add(cur)
            cur = outlet_of.get(cur, 0)
            steps += 1
        if steps >= len(idset) + 5:
            cycles.append(sid)
    if cycles:
        errors.append(f'{len(set(cycles))} segments lie on a routing cycle: '
                      f'{sorted(set(cycles))[:10]}')

    n_outlets = int((segments['outlet'] == 0).sum())
    if n_outlets == 0:
        errors.append('No outlet segment: every segment drains to another.')
    elif n_outlets > 1:
        warnings.append(f'{n_outlets} outlet segments.  This is legitimate when the '
                        f'domain has several independent drainages, but for a single '
                        f'watershed it usually means the mask extends past the '
                        f'true divide.')

    # Reachability: every segment should eventually reach an outlet.
    unreachable = []
    for sid in idset:
        cur, steps = sid, 0
        while cur != 0 and steps < len(idset) + 5:
            cur = outlet_of.get(cur, 0)
            steps += 1
        if steps >= len(idset) + 5:
            unreachable.append(sid)
    if unreachable:
        errors.append(f'{len(unreachable)} segments never reach an outlet.')

    # The routing rank must exceed every upstream rank, or DHSVM's ordered
    # sweep drops the late contributions (see _routingOrder).
    if 'order' in segments.columns:
        rank = dict(zip(segments['ID'].astype(int), segments['order'].astype(int)))
        bad_rank = sum(1 for sid, out in zip(segments['ID'].astype(int),
                                             segments['outlet'].astype(int))
                       if out != 0 and out in rank and rank[out] <= rank[sid])
        if bad_rank:
            errors.append(f'{bad_rank} segments have a routing rank no greater '
                          f'than an upstream segment; channel_route_network '
                          f'would drop their inflow')
        ranks = sorted(set(segments['order'].astype(int)))
        if ranks and ranks != list(range(1, max(ranks) + 1)):
            errors.append('Routing ranks are not contiguous from 1; '
                          'channel_route_network stops at the first gap')

    outlets = None
    if network is not None and terrain is not None and grid is not None:
        outlets = checkOutlets(network, terrain, grid)
        errors.extend(outlets['errors'])
        warnings.extend(outlets['warnings'])

    strahler = (segments['strahler_order'] if 'strahler_order' in segments
                else segments['order'])
    result = dict(ok=(len(errors) == 0), errors=errors, warnings=warnings,
                  n_segments=len(segments), n_outlets=n_outlets,
                  max_order=int(strahler.max()) if len(segments) else 0,
                  max_routing_rank=int(segments['order'].max()) if len(segments) else 0,
                  outlets=outlets)

    if result['ok']:
        logging.info(f"  topology OK: {result['n_segments']:,} segments, "
                     f"{n_outlets} outlet(s), max Strahler order "
                     f"{result['max_order']}, max routing rank "
                     f"{result['max_routing_rank']}")
    else:
        for e in errors:
            logging.error(f'  TOPOLOGY ERROR: {e}')
    for w in warnings:
        logging.warning(f'  topology warning: {w}')
    return result


def compareToReference(stream_mask: np.ndarray,
                       reference: gpd.GeoDataFrame,
                       grid: ModelGrid) -> Dict[str, float]:
    """Compare the derived network against a mapped one (e.g. NHD).

    Reports the drainage density of each and the fraction of derived
    channel cells that fall within one cell of a mapped flowline.  A low
    overlap means the channel-initiation threshold or the burning step
    needs revisiting.
    """
    import rasterio.features
    from scipy import ndimage

    ref_mask = rasterio.features.rasterize(
        [(g, 1) for g in reference.to_crs(grid.crs).geometry if g is not None],
        out_shape=grid.shape, transform=grid.transform,
        fill=0, all_touched=True, dtype='uint8')

    basin = grid.mask if grid.mask is not None else np.ones(grid.shape, 'uint8')
    ref_mask = ref_mask * (basin != 0)

    dilated = ndimage.binary_dilation(ref_mask > 0, iterations=1)
    derived = stream_mask > 0
    overlap = float(np.count_nonzero(derived & dilated)) / max(int(derived.sum()), 1)

    area_km2 = grid.basin_area_km2
    dd_derived = derived.sum() * grid.cellsize / (area_km2 * 1e6) * 1e3
    dd_ref = (ref_mask > 0).sum() * grid.cellsize / (area_km2 * 1e6) * 1e3

    logging.info(f'  derived vs reference network: drainage density '
                 f'{dd_derived:.3f} vs {dd_ref:.3f} km/km^2, '
                 f'{100*overlap:.1f}% of derived cells within 1 cell of a mapped reach')
    return dict(drainage_density_derived=dd_derived,
                drainage_density_reference=dd_ref,
                overlap_fraction=overlap)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

_HEADER = ('###### Generated by WW-DHSVM -- edit with care ######\n'
           '# Generated: {when}\n')


def writeStreamClassFile(filename: str, class_table: pd.DataFrame) -> str:
    """Write ``stream.class.dat``.

    Format, from ``channel_read_classes`` in ``channel.c``::

        ID   width   depth   manning_n   infiltration
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'w') as fid:
        fid.write('#ID   W       D       n        inf\n')
        for _, r in class_table.iterrows():
            fid.write(f"{int(r['ID']):<5d} {r['width']:<7.3f} {r['depth']:<7.3f} "
                      f"{r['manning']:<8.4f} {r['infiltration']:.4f}\n")
    logging.info(f'  wrote {len(class_table)} channel classes -> {filename}')
    return filename


def writeStreamNetworkFile(filename: str, segments: pd.DataFrame) -> str:
    """Write ``stream.network.dat``.

    Format, from ``channel_read_network``::

        ID  order  slope  length  class  outlet_ID  [SAVE  name]
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'w') as fid:
        for _, r in segments.iterrows():
            line = (f"{int(r['ID']):>6d} {int(r['order']):>3d} "
                    f"{r['slope']:>12.6f} {r['length']:>13.5f} "
                    f"{int(r['class']):>4d} {int(r['outlet']):>7d}")
            if bool(r.get('save', False)):
                line += f"   SAVE   \"{r.get('save_name', 'SAVE')}\""
            fid.write(line + '\n')
    n_save = int(segments['save'].sum()) if 'save' in segments else 0
    logging.info(f'  wrote {len(segments)} channel segments '
                 f'({n_save} flagged SAVE) -> {filename}')
    return filename


def writeStreamMapFile(filename: str, stream_map: pd.DataFrame) -> str:
    """Write ``stream.map.dat``.

    Format, from ``channel_grid_read_map``::

        col  row  seg_ID  length  cut_height  cut_width  aspect  [SINK]

    ``col`` and ``row`` are zero-based, row 0 northernmost.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    when = datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')
    with open(filename, 'w') as fid:
        fid.write(_HEADER.format(when=when))
        fid.write('#                   Segment  Cut/Bank     Cut     Segment\n')
        fid.write('#  Col  Row  ID      Length   Height     Width     Aspect   SINK?\n')
        fid.write('#                     (m)      (m)        (m)       (d)    (optional)\n')
        fid.write('#\n')
        # Formatted in bulk: a large basin has O(10^5) records, and
        # iterrows() would dominate the whole setup's runtime.
        fid.writelines(
            f'{c:5d} {r:5d} {s:5d} {l:11.4f} {h:10.4f} {w:9.4f} {a:10.4f}\n'
            for c, r, s, l, h, w, a in zip(
                stream_map['col'].to_numpy(dtype='int64'),
                stream_map['row'].to_numpy(dtype='int64'),
                stream_map['seg_id'].to_numpy(dtype='int64'),
                stream_map['length'].to_numpy(),
                stream_map['cut_height'].to_numpy(),
                stream_map['cut_width'].to_numpy(),
                stream_map['aspect'].to_numpy()))
    logging.info(f'  wrote {len(stream_map)} stream-map records -> {filename}')
    return filename


def writeAll(outdir: str, network: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
    """Write all three DHSVM stream files into ``outdir``."""
    files = {
        'class': writeStreamClassFile(
            os.path.join(outdir, f'{prefix}stream.class.dat'), network['class_table']),
        'network': writeStreamNetworkFile(
            os.path.join(outdir, f'{prefix}stream.network.dat'), network['segments']),
        'map': writeStreamMapFile(
            os.path.join(outdir, f'{prefix}stream.map.dat'), network['map']),
    }
    return files


def writeImperviousRoutingFile(filename: str,
                               channel_mask: np.ndarray,
                               grid: ModelGrid) -> str:
    """Write DHSVM's impervious-surface routing file.

    Whenever **any** vegetation type declares ``Impervious Fraction >
    0``, DHSVM demands this file via the ``IMPERVIOUS SURFACE ROUTING
    FILE`` key in the ``[VEGETATION]`` section (``InitTables.c``).  It
    tells the model where runoff generated on impervious area goes:
    rather than infiltrating or routing downslope cell by cell, that
    water is delivered directly to a nominated channel cell, which is
    what a storm-drain network does in reality.

    Format, from ``InitNetwork.c``: one whitespace-delimited line per
    **in-basin cell, in row-major order**::

        y  x  drains_y  drains_x

    DHSVM re-reads the first two fields and aborts with error 64 if they
    do not match the cell it expects, so the ordering is not optional --
    it must be exactly ``for y in rows: for x in cols: if INBASIN``.

    ``ww_dhsvm`` sets each cell's destination to its nearest channel
    cell by Euclidean distance, which is the same rule PNNL's
    ``find_nearest_channel.c`` applies.

    Parameters
    ----------
    filename : str
        Destination path.
    channel_mask : np.ndarray
        Boolean/int array, non-zero on channel cells.
    grid : ModelGrid
        Supplies the basin mask and shape.

    Returns
    -------
    str
        ``filename``.
    """
    from scipy import ndimage

    mask = grid.mask if grid.mask is not None else np.ones(grid.shape, 'uint8')
    chan = (np.asarray(channel_mask) > 0)
    if not chan.any():
        raise ValueError('Cannot build the impervious routing file: the basin has '
                         'no channel cells.')

    # For every cell, the index of the nearest channel cell.
    _, (ry, rx) = ndimage.distance_transform_edt(~chan, return_indices=True)

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    n = 0
    with open(filename, 'w') as fid:
        for y in range(grid.nrows):
            for x in range(grid.ncols):
                if mask[y, x] == 0:
                    continue
                fid.write(f'{y} {x} {int(ry[y, x])} {int(rx[y, x])}\n')
                n += 1

    logging.info(f'  wrote impervious-surface routing for {n:,} in-basin cells '
                 f'-> {filename}')
    return filename
