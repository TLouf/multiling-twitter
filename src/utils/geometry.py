import numpy as np
from pyproj import Transformer
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry import Point
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


def create_grid(shape_df, cell_size, latlon_proj='epsg:4326',
                xy_proj='epsg:3857', intersect=False):
    '''
    Creates a square grid over a given shape. It is highly preferable to return
    the grid back in (lon, lat) coordinates, because the tweets' coordinates
    are in this coordinate system, and it would much more costly to project all
    these tweets' coordinates to the (x,y) system.
    `shape_df` (GeoDataFrame): single line GeoDataFrame containing the shape on
    which the grid is to be created, in lat,lon coordinates.
    `cell_size` (int): size of the sides of the square cells which constitute
    the grid, in meters.
    `intersect` (bool): determines whether the function computes
    `cells_in_shape_df`, the intersection of the created grid with the shape of
    the area of interest, so that the grid only covers the shape.
    '''
    # We have a one line dataframe, we'll take only the geoseries with the
    # geometry, create a new dataframe out of it with the bounds of this series.
    # Then the values attribute is a one line matrix, so we take the first
    # line and get the bounds' in the format (lon_min, lat_min, lon_max,
    # lat_max)
    lon_min, lat_min, lon_max, lat_max = shape_df['geometry'].bounds.values[0]
    # We create a transformer to project from lon,lat coordinates to x,y.
    # always_xy set to True ensures that we always work with coordinates in the
    # same order: (lon, lat) and (x ,y).
    transformer = Transformer.from_crs(latlon_proj, xy_proj, always_xy=True)
    # Because of curvature, the projection of the point at (lon_min, lat_min)
    # will have a different x coordinate than the point at (lon_min, lat_max),
    # even though those two have the same longitude. Thus, to cover the whole
    # area, the mnimum x we need is the minimum x between these two projections.
    # The same goes for x_max, y_min and y_max.
    x_extrem_list, y_extrem_list = transformer.transform(
        [lon_min, lon_max, lon_min, lon_max],
        [lat_min, lat_min, lat_max, lat_max])
    x_min = min(x_extrem_list)
    x_max = max(x_extrem_list)
    y_min = min(y_extrem_list)
    y_max = max(y_extrem_list)
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
    lon_grid, lat_grid = transformer.transform(x_grid_re, y_grid_re,
                                               direction='INVERSE')
    cells_list = []
    for i in range(Nx-1):
        left_lon = lon_grid[i*Ny]
        right_lon = lon_grid[(i+1)*Ny]
        for j in range(Ny-1):
            bot_lat = lat_grid[j]
            top_lat = lat_grid[j+1]
            # The Polygon closes itself, so no need to repeat the first point at
            # the end.
            cells_list.append(Polygon([
                (left_lon, top_lat), (right_lon, top_lat),
                (right_lon, bot_lat), (left_lon, bot_lat)]))

    cells_df = geopd.GeoDataFrame(cells_list, crs=latlon_proj,
                                  columns=['geometry'])
    cells_df['cell_id'] = cells_df.index
    if intersect:
        cells_in_shape_df = geopd.overlay(
            cells_df, shape_df[['geometry']], how='intersection')
        cells_in_shape_df.index = cells_in_shape_df['cell_id']
    else:
        cells_in_shape_df = None

    return cells_df, cells_in_shape_df, Nx-1, Ny-1


def extract_shape(shape_df, shapefile_name_col, shapefile_name_val,
                  min_area=None, simplify_tol=None, latlon_proj='epsg:4326'):
    '''
    Extracts the shape of the area of interest, which should be located on the
    row where the string in the `shapefile_name_col` of `shape_df` starts with
    `shapefile_name_val`. Then the shape we extract is simplified to accelerate
    later computations, first by removing irrelevant polygons inside the shape
    (if it's comprised of more than one), and then simplifying the contours.
    '''
    shape_df = shape_df.loc[
        shape_df[shapefile_name_col].str.startswith(shapefile_name_val)]
    shape_df = shape_df.to_crs(latlon_proj)
    shapely_geo = shape_df.geometry.iloc[0]

    if min_area is None or simplify_tol is None:
        area_bounds = shapely_geo.bounds
        # Get an upper limit of the distance that can be travelled inside the
        # area
        max_distance = np.sqrt((area_bounds[0]-area_bounds[2])**2
                               + (area_bounds[1]-area_bounds[3])**2)
        if simplify_tol is None:
            simplify_tol = max_distance / 1000

    if type(shapely_geo) == MultiPolygon:
        if min_area is None:
            min_area = max_distance**2 / 1000
        # We delete the polygons in the multipolygon which are too small and
        # just complicate the shape needlessly. The units here are in ° (!)
        shape_df.geometry.iloc[0] = MultiPolygon([poly for poly in shapely_geo
                                                  if poly.area > min_area])
    # We also simplify by a given tolerance (max distance a point can be moved),
    # this could be a parameter in countries.json if needed
    shape_df.geometry = shape_df.simplify(simplify_tol)
    return shape_df



def geo_from_bbox(bbox):
    '''
    From a bounding box dictionary `bbox`, which is of the form
    {'coordinates': [[[x,y] at top right, [x,y] at top left, ...]]}, returns the
    geometry and its area. If the four coordinates are actually the same, they
    define a null-area Polygon, or rather something better defined as a Point.
    '''
    bbox = bbox['coordinates'][0]
    geo = Polygon(bbox)
    area = geo.area
    if area == 0:
        geo = Point(bbox[0])
    return geo, area


def make_places_geodf(places_df, shape_df, latlon_proj='epsg:4326',
                      xy_proj='epsg:3857'):
    '''
    Constructs a GeoDataFrame with all the places in `places_df` which have
    their centroid within `shape_df`, and calculates their area within the shape
    in squared meters.
    '''
    places_df['geometry'], places_df['area'] = zip(
        *places_df['bounding_box'].apply(geo_from_bbox))
    places_geodf = geopd.GeoDataFrame(
        places_df, crs=latlon_proj, geometry=places_df['geometry'])
    places_geodf = places_geodf.set_index('id', drop=False)
    # We then get the places centroids to check that they are within our area
    # (useful when we're only interested in the region of a country).
    places_centroids = geopd.GeoDataFrame(
        geometry=places_geodf[['geometry']].centroid, crs=latlon_proj)
    places_in_shape = geopd.sjoin(places_centroids, shape_df[['geometry']],
                                  op='within', rsuffix='shape')
    places_geodf = places_geodf.join(places_in_shape['index_shape'],
                                     how='inner')
    # Since the places' bbox can stretch outside of the whole shape, we need to
    # take the intersection between the two. However, we only use the overlay to
    # calculate the area, so that places_to_cells distributes the whole
    # population of a place to the different cells within it. However we don't
    # need the actual geometry from the intersection, which is more complex and
    # thus slows down computations later on.
    poly_mask = places_geodf['area'] > 0
    polygons_in_shape = geopd.overlay(
        shape_df[['geometry']], places_geodf.loc[poly_mask], how='intersection')
    polygons_in_shape = polygons_in_shape.set_index('id')
    places_geodf.loc[poly_mask, 'area'] = polygons_in_shape.to_crs(xy_proj).area
    places_geodf = places_geodf.drop(
        columns=['bounding_box', 'id','index_shape'])
    places_in_xy = places_geodf.geometry.to_crs(xy_proj)
    return places_geodf, places_in_xy
