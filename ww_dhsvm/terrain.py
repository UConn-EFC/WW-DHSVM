"""Terrain preprocessing for DHSVM.

DHSVM routes water on the same regular grid it solves on: saturated
subsurface flow follows the water-table (or topographic) gradient between
neighbouring cells, and overland flow follows the surface gradient.  That
makes the conditioned DEM the single most influential input in the whole
setup -- a sink left in the DEM becomes a permanent lake, and a flat area
becomes a place where the routing gradient is zero and water stops
moving.

This module therefore does four things:

1. **Resample** a source DEM (3DEP by default) onto the model grid.
2. **Condition** it hydrologically -- fill pits, resolve flats, and
   optionally burn a reference stream network in.
3. **Derive** slope, aspect, flow direction and flow accumulation.
4. **Estimate** soil depth from terrain.

Watershed Workflow performs the analogous conditioning on an unstructured
mesh (``condition.fillPitsDual``); here everything is raster algebra, so
``pyflwdir`` -- which implements the standard Wang & Liu (2006) depression
filling and D8 flow routing on regular grids -- does the heavy lifting.

Note on DHSVM's routing stencil
-------------------------------
DHSVM's default surface/subsurface routing is **D4**: a cell distributes
outflow to all four down-gradient von Neumann neighbours, weighted by
gradient.  The model can also be compiled for **D8** (``Routing
Neighbors = 8``), which sends all outflow to the single steepest of the
eight neighbours.  The flow-direction and accumulation grids computed
here are D8 -- they are used to *delineate the channel network*, not to
route water at runtime, and D8 is the right choice for channel
delineation regardless of which stencil DHSVM later uses.
"""

from typing import Optional, Tuple, Dict, Any
import logging

import numpy as np
import xarray as xr
import shapely.geometry
import geopandas as gpd
import rasterio.features
import rasterio.warp
import rasterio.enums

import ww_dhsvm.crs
import ww_dhsvm.warp
from ww_dhsvm.grid import ModelGrid


def gridFromRaster(path: str) -> ModelGrid:
    """Build a :class:`ModelGrid` from an existing north-up raster.

    Bring-your-own-DEM entry: the raster's extent, cell size and CRS
    become the model grid, so a DEM prepared elsewhere (a clipped and
    reprojected tile, a lidar product, a colleague's grid) is used as is
    instead of being fetched and resampled.  The raster must be square-
    celled, north-up and in a projected CRS.

    Parameters
    ----------
    path : str
        A raster readable by rasterio.

    Returns
    -------
    ModelGrid
        With no mask; see :func:`demFromRaster` for the mask.
    """
    import rasterio

    with rasterio.open(path) as src:
        tr = src.transform
        if abs(tr.b) > 0 or abs(tr.d) > 0 or tr.e >= 0:
            raise ValueError(f'{path}: the raster must be north-up (no rotation)')
        if abs(abs(tr.a) - abs(tr.e)) > 1e-6 * abs(tr.a):
            raise ValueError(f'{path}: cells must be square, got {tr.a} x {tr.e}')
        if src.crs is None:
            raise ValueError(f'{path}: the raster has no CRS')
        crs = ww_dhsvm.crs.from_rasterio(src.crs)
        if not crs.is_projected:
            raise ValueError(f'{path}: the raster CRS must be projected (metres)')
        grid = ModelGrid(src.height, src.width, abs(tr.a),
                         src.bounds.left, src.bounds.bottom, crs)
    logging.info(f'  model grid from {path}: {grid.nrows} rows x {grid.ncols} cols '
                 f'@ {grid.cellsize:g} m')
    return grid


def demFromRaster(path: str,
                  grid: Optional[ModelGrid] = None,
                  mask_from_nodata: bool = True,
                  fill: bool = True) -> Tuple[ModelGrid, np.ndarray]:
    """Read a DEM from an existing raster onto the model grid.

    Two ways to use it.  Without ``grid``, the raster defines the grid
    (see :func:`gridFromRaster`) and, if it is clipped to the basin, its
    no-data cells define the basin mask.  With ``grid``, the raster is
    read through the window that covers the grid, which must lie on the
    raster's cell edges; this is how a basin-clipped DEM (grid and mask)
    and the unclipped tile it was cut from (real elevations outside the
    mask, which the rectangle-plus-mask convention of
    :func:`conditionDEM` expects) are combined.

    Parameters
    ----------
    path : str
        A raster readable by rasterio, in the grid's CRS.
    grid : ModelGrid, optional
        The target grid.  When ``None`` it is built from the raster.
    mask_from_nodata : bool, optional
        When the grid has no mask yet, take the raster's valid cells as
        the basin mask.  Default ``True``.
    fill : bool, optional
        Fill no-data cells by nearest-neighbour dilation
        (:func:`fillGaps`) so the array is NaN-free, as
        :func:`conditionDEM` requires.  Default ``True``.

    Returns
    -------
    (ModelGrid, np.ndarray)
        The grid (with its mask set when requested) and the float64 DEM.
    """
    import rasterio
    import rasterio.windows

    if grid is None:
        grid = gridFromRaster(path)
    with rasterio.open(path) as src:
        src_crs = ww_dhsvm.crs.from_rasterio(src.crs)
        if not ww_dhsvm.crs.isEqual(src_crs, grid.crs):
            raise ValueError(f'{path}: CRS {ww_dhsvm.crs.toString(src_crs)} differs '
                             f'from the grid CRS {ww_dhsvm.crs.toString(grid.crs)}')
        if abs(abs(src.transform.a) - grid.cellsize) > 1e-6 * grid.cellsize:
            raise ValueError(f'{path}: cell size {abs(src.transform.a)} differs from '
                             f'the grid cell size {grid.cellsize}')
        win = rasterio.windows.from_bounds(*grid.bounds, transform=src.transform)
        col0, row0 = win.col_off, win.row_off
        if (abs(col0 - round(col0)) > 1e-3 or abs(row0 - round(row0)) > 1e-3
                or abs(win.width - grid.ncols) > 1e-3
                or abs(win.height - grid.nrows) > 1e-3):
            raise ValueError(f'{path}: the grid does not lie on this raster\'s cell '
                             f'edges (window offset {col0:.3f}, {row0:.3f}; size '
                             f'{win.width:.3f} x {win.height:.3f})')
        win = rasterio.windows.Window(int(round(col0)), int(round(row0)),
                                      grid.ncols, grid.nrows)
        arr = src.read(1, window=win, boundless=True,
                       fill_value=src.nodata if src.nodata is not None else np.nan)
        arr = arr.astype('float64')
        if src.nodata is not None and np.isfinite(src.nodata):
            arr[arr == src.nodata] = np.nan
    valid = np.isfinite(arr)
    if mask_from_nodata and grid.mask is None:
        grid.mask = valid.astype('uint8')
        n = int(grid.mask.sum())
        logging.info(f'  basin mask from the raster\'s valid cells: {n} of '
                     f'{grid.nrows * grid.ncols} ({n * grid.cell_area / 1e6:.2f} km^2)')
    if fill and not valid.all():
        arr = fillGaps(arr)
    logging.info(f'  DEM from {path}: {np.nanmin(arr):.1f} to {np.nanmax(arr):.1f} m, '
                 f'{int(valid.sum())} valid cells')
    return grid, arr


def resampleToGrid(dataset,
                   grid: ModelGrid,
                   resampling: str = 'bilinear',
                   nodata: float = np.nan) -> np.ndarray:
    """Resample a georeferenced raster onto the model grid.

    Parameters
    ----------
    dataset : xr.DataArray or xr.Dataset
        Source raster.  Must carry rioxarray CRS/transform metadata.  If
        a Dataset, it must hold exactly one data variable.
    grid : ModelGrid
        Destination grid.
    resampling : str, optional
        A ``rasterio.enums.Resampling`` name.  Use ``'bilinear'`` or
        ``'cubic'`` for continuous fields (elevation, soil fraction) and
        ``'nearest'`` or ``'mode'`` for categorical ones (land cover,
        soil class).  Default ``'bilinear'``.
    nodata : float, optional
        Fill value for destination cells with no source coverage.

    Returns
    -------
    np.ndarray
        2D array shaped ``grid.shape``, row 0 in the north.
    """
    if isinstance(dataset, xr.Dataset):
        varnames = list(dataset.data_vars)
        if len(varnames) != 1:
            raise ValueError(f'resampleToGrid needs one variable, got {varnames}')
        dataset = dataset[varnames[0]]

    src_crs = dataset.rio.crs
    if src_crs is None:
        raise ValueError('Source raster has no CRS; cannot resample.')

    src = np.asarray(dataset.values, dtype='float64')
    if src.ndim == 3 and src.shape[0] == 1:
        src = src[0]
    if src.ndim != 2:
        raise ValueError(f'Expected a 2D raster, got shape {src.shape}')

    src_nodata = dataset.rio.nodata
    if src_nodata is not None and not np.isnan(src_nodata):
        src = np.where(src == src_nodata, np.nan, src)

    dst = np.full(grid.shape, np.nan, dtype='float64')
    rasterio.warp.reproject(
        source=src,
        destination=dst,
        src_transform=dataset.rio.transform(),
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=getattr(rasterio.enums.Resampling, resampling))

    n_missing = int(np.count_nonzero(np.isnan(dst)))
    if n_missing:
        logging.info(f'  resample: {n_missing} of {dst.size} destination cells '
                     f'have no source coverage')
        if not np.isnan(nodata):
            dst = np.where(np.isnan(dst), nodata, dst)
    return dst


def fillGaps(array: np.ndarray, mask: Optional[np.ndarray] = None,
             max_iterations: int = 200) -> np.ndarray:
    """Fill NaN holes by iterative nearest-neighbour dilation.

    A DEM clipped from a tiled source occasionally has small no-data
    holes.  DHSVM's binary maps have no nodata convention, so every
    in-basin cell must carry a real value.  This grows valid values into
    the holes one ring at a time, which preserves local means far better
    than substituting a global constant.

    Parameters
    ----------
    array : np.ndarray
        2D array possibly containing NaN.
    mask : np.ndarray, optional
        Only cells where ``mask`` is non-zero need to be filled.
    max_iterations : int, optional
        Safety bound on the dilation loop.

    Returns
    -------
    np.ndarray
        Copy of ``array`` with holes filled.
    """
    from scipy import ndimage

    out = np.array(array, dtype='float64', copy=True)
    need = np.isnan(out)
    if mask is not None:
        need &= (mask != 0)
    if not need.any():
        return out

    logging.info(f'  filling {int(need.sum())} no-data cells by nearest-neighbour dilation')
    valid = ~np.isnan(out)
    if not valid.any():
        raise ValueError('Array is entirely no-data; nothing to fill from.')

    # distance_transform_edt with return_indices gives, for every cell,
    # the index of the nearest valid cell -- one pass, no iteration needed.
    _, (ri, ci) = ndimage.distance_transform_edt(~valid, return_indices=True)
    out[need] = out[ri[need], ci[need]]
    return out


def conditionDEM(dem: np.ndarray,
                 grid: ModelGrid,
                 streams: Optional[gpd.GeoDataFrame] = None,
                 burn_depth: float = 5.0,
                 outlets: Optional[np.ndarray] = None,
                 method: str = 'wang_liu',
                 confine_to_basin: bool = True) -> Dict[str, Any]:
    """Hydrologically condition a DEM and derive flow-routing grids.

    Parameters
    ----------
    dem : np.ndarray
        Raw elevation on the model grid, metres.  Must be NaN-free.
    grid : ModelGrid
        The model grid.
    streams : gpd.GeoDataFrame, optional
        A reference stream network (e.g. NHD flowlines).  When given, the
        DEM is *burned*: elevations along these lines are lowered by
        ``burn_depth`` before filling, which forces the derived channel
        network to follow mapped rivers.  This matters most in low-relief
        basins where a 10--30 m DEM does not resolve the true channel.
    burn_depth : float, optional
        Metres to lower burned cells.  Default 5.0.
    outlets : np.ndarray, optional
        Boolean array marking cells that may drain off-grid.  Defaults to
        the grid edge.
    method : str, optional
        Depression-handling method.  ``'wang_liu'`` (default) fills
        depressions to their spill elevation.
    confine_to_basin : bool, optional
        Compute flow directions only within ``grid.mask``.  Default
        ``True``, and it matters: the model grid is a *rectangle* around
        the watershed, so parts of neighbouring basins sit inside it, and
        an unconfined D8 field lets them drain across the divide.  On the
        Connecticut that inflated the largest contributing area to
        35,127 km^2 against a true basin area of 29,186 km^2 -- **20%
        of the main stem's drainage arriving from outside the watershed**,
        which then propagates into channel width and depth through
        hydraulic geometry.  DHSVM simulates only in-basin cells and
        applies no boundary inflow, so confining the routing is what
        makes the delineation consistent with what the model will do.
        Set ``False`` only when the mask is deliberately larger than the
        contributing area.

    Returns
    -------
    dict
        With keys ``'dem'`` (conditioned elevation), ``'flowdir'`` (D8
        directions in pyflwdir encoding), ``'flowacc'`` (upstream cell
        count), ``'uparea'`` (upstream area, m^2), ``'slope'`` (radians),
        ``'aspect'`` (degrees clockwise from north), ``'n_pits_filled'``
        and ``'fill_volume'``.
    """
    import pyflwdir

    if np.any(np.isnan(dem)):
        raise ValueError('conditionDEM requires a NaN-free DEM; call fillGaps first.')

    work = np.array(dem, dtype='float32', copy=True)

    # --- optional stream burning ------------------------------------
    if streams is not None and len(streams) > 0:
        burn_mask = rasterio.features.rasterize(
            [(g, 1) for g in streams.to_crs(grid.crs).geometry if g is not None],
            out_shape=grid.shape, transform=grid.transform,
            fill=0, all_touched=True, dtype='uint8')
        n_burn = int(burn_mask.sum())
        if n_burn:
            work[burn_mask > 0] -= burn_depth
            logging.info(f'  burned {n_burn} cells along the reference stream '
                         f'network by {burn_depth:g} m')

    # --- depression filling -----------------------------------------
    pre = work.copy()
    filled = pyflwdir.dem.fill_depressions(work, outlets='edge' if outlets is None else 'min')[0]
    diff = filled - pre
    n_pits = int(np.count_nonzero(diff > 1e-6))
    fill_volume = float(np.sum(diff[diff > 0]) * grid.cell_area)
    logging.info(f'  depression filling ({method}): raised {n_pits} cells, '
                 f'{fill_volume/1e6:.3f} x 10^6 m^3 of fill, '
                 f'max raise {float(diff.max()):.2f} m')

    # --- D8 flow direction and accumulation -------------------------
    # Exclude out-of-basin cells by making them nodata, so no flow path
    # can enter across the divide.
    routing_dem = filled
    if confine_to_basin and grid.mask is not None:
        routing_dem = np.where(grid.mask != 0, filled, np.nan).astype('float32')
        n_out = int(np.count_nonzero(grid.mask == 0))
        logging.info(f'  confining flow routing to the basin: {n_out:,} '
                     f'out-of-basin cells excluded so nothing drains across '
                     f'the divide')

    flw = pyflwdir.from_dem(data=routing_dem, nodata=np.nan,
                            transform=grid.transform, latlon=False)
    flowacc = flw.upstream_area(unit='cell')
    flowacc = np.where(np.isfinite(flowacc), flowacc, 0)
    uparea = np.where(flowacc > 0, flowacc * grid.cell_area, grid.cell_area)

    # --- slope and aspect -------------------------------------------
    slope, aspect = slopeAspect(filled, grid.cellsize)

    logging.info(f'  slope: mean {np.degrees(np.nanmean(slope)):.2f} deg, '
                 f'max {np.degrees(np.nanmax(slope)):.2f} deg')
    max_ua = float(np.nanmax(uparea)) / 1e6
    logging.info(f'  max upstream area: {max_ua:,.0f} km^2')
    if grid.mask is not None and grid.basin_area_km2 > 0:
        ratio = max_ua / grid.basin_area_km2
        if ratio > 1.02:
            logging.warning(
                f'  largest contributing area is {100*(ratio-1):.0f}% above the '
                f'basin area ({grid.basin_area_km2:,.0f} km^2) -- flow is '
                f'entering across the divide')

    return dict(dem=filled, raw_dem=dem, flowdir=flw, flowacc=flowacc,
                uparea=uparea, slope=slope, aspect=aspect,
                n_pits_filled=n_pits, fill_volume=fill_volume,
                fill_amount=diff)


def slopeAspect(dem: np.ndarray, cellsize: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute slope and aspect with Horn's third-order finite difference.

    Horn (1981) is the method ArcGIS and GRASS use, and is what the
    legacy DHSVM AML preprocessing assumed, so using it here keeps
    ``ww_dhsvm``'s soil-depth estimates comparable to setups built with
    the original toolchain.

    Parameters
    ----------
    dem : np.ndarray
        Elevation, metres, row 0 in the north.
    cellsize : float
        Grid spacing, metres.

    Returns
    -------
    slope : np.ndarray
        Slope in **radians**.
    aspect : np.ndarray
        Aspect in **degrees clockwise from north**, in ``[0, 360)``.
        Flat cells are assigned an aspect of 0.
    """
    z = np.pad(np.asarray(dem, dtype='float64'), 1, mode='edge')

    # Horn's kernel; note row index increases southward, so dz/dy is
    # negated relative to a north-up mathematical y axis.
    dzdx = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) -
            (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8.0 * cellsize)
    dzdy = ((z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:]) -
            (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])) / (8.0 * cellsize)

    slope = np.arctan(np.hypot(dzdx, dzdy))

    # Aspect: direction of steepest descent, degrees clockwise from north.
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = np.where(aspect < 0, 90.0 - aspect,
                      np.where(aspect > 90.0, 360.0 - aspect + 90.0, 90.0 - aspect))
    aspect = np.mod(aspect, 360.0)
    aspect = np.where(np.hypot(dzdx, dzdy) < 1e-12, 0.0, aspect)
    return slope, aspect


def estimateSoilDepth(slope: np.ndarray,
                      elevation: np.ndarray,
                      uparea: np.ndarray,
                      min_depth: float = 0.5,
                      max_depth: float = 3.0,
                      wt_slope: float = 0.7,
                      wt_source: float = 0.0,
                      wt_elev: float = 0.3,
                      max_slope_deg: float = 30.0,
                      max_source: float = 1.0e5,
                      max_elev: float = 1500.0,
                      pow_slope: float = 0.25,
                      pow_source: float = 1.0,
                      pow_elev: float = 0.75,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate soil depth from terrain, following PNNL's ``soildepth.aml``.

    DHSVM needs a total soil-column depth for every cell, and no national
    dataset provides it directly.  The DHSVM community's standard
    substitute is a weighted terrain index: soils are deep in flat, low,
    high-accumulation positions and thin on steep, high, divergent ones.

        depth = min + (max - min) * [ w_s (1 - (S/S_max)^p_s)
                                    + w_a (A/A_max)^p_a
                                    + w_e (1 - (E/E_max)^p_e) ]

    with each driver clipped at its maximum before the power is applied,
    and the three weights summing to 1.  Defaults reproduce PNNL's
    ``soildepthscript.py``.

    This is a *terrain proxy, not a measurement*.  Where SoilGrids or
    Pelletier depth-to-bedrock is credible for a basin, prefer it; this
    function exists because those products are frequently implausible at
    DHSVM's resolution.  See :func:`blendSoilDepth`.

    Parameters
    ----------
    slope : np.ndarray
        Slope in radians.
    elevation : np.ndarray
        Elevation, metres.
    uparea : np.ndarray
        Upstream contributing area, m^2.
    min_depth, max_depth : float, optional
        Floor and ceiling on the result, metres.
    wt_slope, wt_source, wt_elev : float, optional
        Relative weights; must sum to 1.
    max_slope_deg, max_source, max_elev : float, optional
        Saturation values for each driver.
    pow_slope, pow_source, pow_elev : float, optional
        Exponents shaping each driver's response.
    mask : np.ndarray, optional
        Cells to compute; elsewhere ``min_depth`` is returned.

    Returns
    -------
    np.ndarray
        Soil depth, metres.
    """
    total = wt_slope + wt_source + wt_elev
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f'Soil-depth weights must sum to 1.0, got {total}')

    slope_deg = np.degrees(np.asarray(slope, dtype='float64'))
    s = np.clip(slope_deg, 0.0, max_slope_deg) / max_slope_deg
    a = np.clip(np.asarray(uparea, dtype='float64'), 0.0, max_source) / max_source
    e = np.clip(np.asarray(elevation, dtype='float64'), 0.0, max_elev) / max_elev

    frac = (wt_slope * (1.0 - np.power(s, pow_slope))
            + wt_source * np.power(a, pow_source)
            + wt_elev * (1.0 - np.power(e, pow_elev)))
    depth = min_depth + (max_depth - min_depth) * np.clip(frac, 0.0, 1.0)

    if mask is not None:
        depth = np.where(mask != 0, depth, min_depth)

    logging.info(f'  soil depth (terrain index): {depth.min():.2f}--{depth.max():.2f} m, '
                 f'mean {depth.mean():.2f} m')
    return depth


def blendSoilDepth(terrain_depth: np.ndarray,
                   observed_depth: Optional[np.ndarray],
                   weight_observed: float = 0.5,
                   min_depth: float = 0.5,
                   max_depth: float = 3.0) -> np.ndarray:
    """Blend the terrain soil-depth proxy with an observed product.

    SoilGrids' and Pelletier's depth-to-bedrock layers carry real
    information about regional regolith thickness but are far too coarse
    (250 m--1 km) to capture the hillslope-scale variation DHSVM's
    subsurface routing responds to.  The terrain index has the opposite
    problem.  A convex combination keeps the regional signal from the
    observation and the hillslope structure from the terrain.

    Parameters
    ----------
    terrain_depth : np.ndarray
        Output of :func:`estimateSoilDepth`.
    observed_depth : np.ndarray or None
        An observed depth-to-bedrock raster already on the model grid,
        metres.  When ``None`` the terrain estimate is returned unchanged.
    weight_observed : float, optional
        Weight on the observation, in ``[0, 1]``.  Default 0.5.
    min_depth, max_depth : float, optional
        Clip bounds applied after blending.

    Returns
    -------
    np.ndarray
        Blended soil depth, metres.
    """
    if observed_depth is None:
        return terrain_depth

    obs = np.where(np.isnan(observed_depth), terrain_depth, observed_depth)
    w = float(np.clip(weight_observed, 0.0, 1.0))
    out = np.clip((1.0 - w) * terrain_depth + w * obs, min_depth, max_depth)
    logging.info(f'  soil depth blended {100*(1-w):.0f}% terrain / {100*w:.0f}% observed: '
                 f'{out.min():.2f}--{out.max():.2f} m, mean {out.mean():.2f} m')
    return out


def enforceSoilDepthBelowChannels(soil_depth: np.ndarray,
                                  cut_height: np.ndarray,
                                  margin: float = 1.05) -> np.ndarray:
    """Guarantee soil depth exceeds channel cut depth everywhere.

    ``channel_grid.c`` rejects any stream-map record whose ``cut_height``
    exceeds the cell's soil depth, and merely warns-and-overrides in
    other places.  Raising the soil column to at least ``margin`` times
    the local cut depth removes a whole class of startup failures.

    Parameters
    ----------
    soil_depth : np.ndarray
        Soil depth, metres.
    cut_height : np.ndarray
        Per-cell channel cut depth, metres; zero off-channel.
    margin : float, optional
        Safety factor.  Default 1.05.

    Returns
    -------
    np.ndarray
        Adjusted soil depth.
    """
    need = cut_height * margin
    out = np.maximum(soil_depth, need)
    n = int(np.count_nonzero(out > soil_depth + 1e-9))
    if n:
        logging.info(f'  raised soil depth in {n} channel cells to stay below '
                     f'the channel bed (margin {margin:g}x)')
    return out


def maskedStats(array: np.ndarray, mask: Optional[np.ndarray], name: str = '') -> Dict[str, float]:
    """Summary statistics over in-basin cells, for logging and QA."""
    a = np.asarray(array, dtype='float64')
    if mask is not None:
        a = a[mask != 0]
    a = a[~np.isnan(a)]
    if a.size == 0:
        return dict(name=name, n=0)
    return dict(name=name, n=int(a.size), min=float(a.min()), max=float(a.max()),
                mean=float(a.mean()), std=float(a.std()),
                p05=float(np.percentile(a, 5)), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)))


def enforceSoilDepthBelowRootZone(soil_depth: np.ndarray,
                                  veg_class: np.ndarray,
                                  veg_blocks,
                                  mask: Optional[np.ndarray] = None,
                                  margin: float = 1.10) -> np.ndarray:
    """Guarantee soil depth exceeds the local total rooting depth.

    ``CheckOut.c`` treats ``SoilMap[y][x].Depth <= VType[...].TotalDepth``
    as a **fatal** error, where ``TotalDepth`` is the sum of that
    vegetation type's ``Root Zone Depths`` (which are layer
    *thicknesses*, not cumulative depths).  Physically the requirement is
    obvious -- roots cannot extend below the soil column -- but the
    terrain-index soil depth of :func:`estimateSoilDepth` knows nothing
    about vegetation, so on steep cells under deep-rooted forest the two
    can conflict.

    This raises soil depth wherever needed, by vegetation type, to
    ``margin`` times the local rooting depth.

    Parameters
    ----------
    soil_depth : np.ndarray
        Soil depth, metres.
    veg_class : np.ndarray
        DHSVM vegetation IDs on the grid.
    veg_blocks : list of dict
        From :func:`ww_dhsvm.vegetation.buildVegetationBlocks`.
    mask : np.ndarray, optional
        Basin mask; only in-basin cells are adjusted.
    margin : float, optional
        Safety factor above the rooting depth.  Default 1.10.

    Returns
    -------
    np.ndarray
        Adjusted soil depth.
    """
    required = np.zeros(256, dtype='float64')
    for b in veg_blocks:
        required[b['id']] = float(np.sum(b['root_zone_depths'])) * margin

    need = required[veg_class]
    if mask is not None:
        need = np.where(mask != 0, need, 0.0)

    out = np.maximum(np.asarray(soil_depth, dtype='float64'), need)
    n = int(np.count_nonzero(out > np.asarray(soil_depth) + 1e-9))
    if n:
        deepest = float(required.max())
        logging.info(f'  raised soil depth in {n:,} cells to stay below the root '
                     f'zone (deepest requirement {deepest:.2f} m, margin {margin:g}x)')
    return out
