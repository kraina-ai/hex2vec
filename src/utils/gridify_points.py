"""
This comes from https://github.com/mthh/gpd_lite_toolbox/blob/ce4553c139775953c49d8756960d0a166b67c6bf/gpd_lite_toolbox/core.py#L637

"""
import numpy as np






def gridify_data(gdf, height, col_name, cut=True, method=np.mean):
    """
    Taken directly from 
    
    This comes from https://github.com/mthh/gpd_lite_toolbox/blob/ce4553c139775953c49d8756960d0a166b67c6bf/gpd_lite_toolbox/core.py#L637


    Gridify a collection of point observations.
    Parameters
    ----------
    gdf: GeoDataFrame
        The collection of polygons to be covered by the grid.
    height: Integer
        The dimension (will be used as height and width) of the ceils to create,
        in units of *gdf*.
    col_name: String
        The name of the column containing the value to use for the grid cells.
    cut: Boolean, default True
        Cut the grid to fit the shape of *gdf* (ceil partially covering it will
        be truncated). If False, the returned grid fit the bounding box of gdf.
    method: Numpy/Pandas function
        The method to aggregate values of points for each cell.
        (like numpy.max, numpy.mean, numpy.mean, numpy.std or numpy.sum)
    Returns
    -------
    grid: GeoDataFrame
        A collection of polygons.
    Example
    -------
    >>> all(gdf.geometry.type == 'Point')  # The function only act on Points
    True
    >>> gdf.time.dtype  # And the value to aggreagate have to be numerical
    dtype('int64')
    >>> grid_data = gridify_data(gdf, 7500, 'time', method=np.min)
    >>> plot_dataframe(grid_data, column='time')
    <matplotlib.axes._subplots.AxesSubplot at 0x7f8336373a20>
    ...
    """
    if not all(gdf.geometry.type == 'Point'):
        raise ValueError("Can only gridify scattered data (Point)")
    if not gdf[col_name].dtype.kind in {'i', 'f'}:
        raise ValueError("Target column have to be a numerical field")

    grid = make_grid(gdf, height, cut)
    grid[col_name] = -1
    index = make_index([i.bounds for i in gdf.geometry])
    for id_cell in range(len(grid)):
        ids_pts = list(index.intersection(
            grid.geometry[id_cell].bounds, objects='raw'))
        if ids_pts:
            res = method(gdf.iloc[ids_pts][col_name])
            grid.loc[id_cell, col_name] = res
    return grid