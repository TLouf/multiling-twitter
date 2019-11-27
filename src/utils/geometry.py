import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon
import geopandas as geopd

def haversine(lon1, lat1, lon2, lat2):
    R = 6367
    lon1 = lon1 * np.pi / 180
    lat1 = lat1 * np.pi / 180
    lon2 = lon2 * np.pi / 180
    lat2 = lat2 * np.pi / 180

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    d = R * c
    return d


def create_grid(shape_df, cell_size, latlon_proj, xy_proj, intersect=False):
    '''
    Creates a square grid over a given shape.
    shape (GeoDataFrame): single line GeoDataFrame containing the shape on which
    the grid is to be created, in lat,lon coordinates.
    cell_size (int): size of the sides of the square cells which constitute the grid
    latlon_proj (str):
    xy_proj (str):
    intersect (bool): determines whether the function computes
    'cells_in_shape_df', the intersection of the created grid with the shape of
    the object, so that the grid only covers the shape.
    '''
    # We have a one line dataframe, we'll take only the geoseries with the
    # geometry, create a new dataframe out of it with the bounds of this series.
    # Then the values attribute is a one line matrix, so we take the first
    # line and get the bounds' in the format (lon_min, lat_min, lon_max,
    # lat_max)
    lon_min, lat_min, lon_max, lat_max = shape_df['geometry'].bounds.values[0]
    crs = {'init': latlon_proj}
    # We create a transformer to project from lon,lat coordinates to x,y.
    # always_xy set to True ensures that we always work with coordinates in the
    # same order: (lon, lat) and (x ,y).
    transformer = Transformer.from_crs(latlon_proj, xy_proj, always_xy=True)
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    # We want to cover at least the whole shape of the area, because if we want
    # to restrict just to the shape we can then intersect the grid with its
    # shape. Hence the x,ymax+cell_size in the arange.
    x_grid = np.arange(x_min, x_max+cell_size, cell_size)
    y_grid = np.arange(y_min, y_max+cell_size, cell_size)
    Nx = len(x_grid)
    Ny = len(y_grid)
    # We want (x1 x1 ...(Ny times) x1 x2 x2 ...  xNx) and (y1 y2 ... yNy ...(Nx
    # times)), so we use repeat and tile:
    x_grid_re = np.repeat(x_grid, Ny)
    y_grid_re = np.tile(y_grid, Nx)
    # So then (x_grid_re[i], y_grid_re[i]) for all i are all the edges in the
    # grid, which are the points to transform, as requested by the transform
    # method.
    lon_grid, lat_grid = transformer.transform(x_grid_re, y_grid_re, direction='INVERSE')
    cells_list = []
    for i in range(Nx-1):
        left_lon = lon_grid[i*Ny]
        right_lon = lon_grid[(i+1)*Ny]
        for j in range(Ny-1):
            bot_lat = lat_grid[j]
            top_lat = lat_grid[j+1]
            # The Polygon closes itself, so no need to repeat the first point at
            # the end
            cells_list.append(Polygon([
                (left_lon, top_lat), (right_lon, top_lat),
                (right_lon, bot_lat), (left_lon, bot_lat)]))

    cells_df = geopd.GeoDataFrame(cells_list, crs=crs, columns=['geometry'])
    if intersect:
        cells_in_shape_df = geopd.overlay(
            cells_df, shape_df, how='intersection')
    else:
        cells_in_shape_df = None

    return cells_df, cells_in_shape_df
