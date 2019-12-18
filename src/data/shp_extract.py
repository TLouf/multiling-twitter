import pandas as pd
import geopandas as geopd
import numpy as np

def get_cities_geometry(datafile_path, shapefile_path, filters, init_cols,
        data_id_col, shp_id_col, final_cols=None, init_proj='epsg:4326',
        csv_engine=None):
    '''
    Returns a geo dataframe containing the shapes contained in the file located
    at 'shapefile_path', which match every condition set by the list of
    functions 'filters', depending on the data contained in the csv file in
    'datafile_path'.
    '''
    if final_cols is None:
        final_cols = init_cols
    # We read the csv file containing the data and convert it into a dataframe,
    # using the 'python' engine, which is less efficient but supports parsing
    # fields which are in quotations marks.
    data_df = pd.read_csv(datafile_path, usecols=init_cols, engine=csv_engine)
    evaluated_filters = [filter_fun(data_df) for filter_fun in filters]
    # Then create a mask to extract only the lines in data_df which match the
    # filters.
    mask = np.logical_and.reduce(evaluated_filters)
    data_df = data_df.loc[mask, final_cols]
    geo_df = geopd.read_file(shapefile_path)
    geo_col = geo_df.geometry.name
    geo_df = geo_df.loc[:, [shp_id_col, geo_col]]
    geo_df.rename(columns={shp_id_col: data_id_col}, inplace=True)
    # The following was added to get rid of trailing zeros, which appeared in a
    # Spanish dataset for instance. Another solution, if other issues arise, is
    # to add a pre processing function for the ID column as an argument
    try:
        data_df[data_id_col] = data_df[data_id_col].astype('int64')
    except ValueError:
        pass
    id_type = data_df[data_id_col].dtype
    try:
        geo_df[data_id_col] = geo_df[data_id_col].astype(id_type)
    except ValueError:
        print("The IDs can't match because of their types, please check the data")
    geo_df = geo_df.set_index(data_id_col)
    data_df = data_df.set_index(data_id_col)
    final_area_df = geo_df.join(data_df, how='inner')
    final_area_df.crs = {'init': init_proj}
    return final_area_df
