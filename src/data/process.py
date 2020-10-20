import pandas as pd
import geopandas as geopd
import numpy as np
from shapely.geometry import Point
import src.data.text_process as text_process


def post_multi(results, min_length=10000):
    '''
    From a list of dataframes `results` returned by a pool of workers, returns
    the same list but with big enough dataframes. Thus when a dataframe is too
    small, it is concatenated to another one.
    '''
    # We first get rid of the null results.
    results = [res for res in results if res is not None]
    df_sizes = pd.Series([df.shape[0] for df in results], name='length')
    # The following series will have as index the indices in `results` of the
    # dataframes which are too small, and that we will concatenate to another.
    # The index is ordered by descending length.
    to_concat = df_sizes.loc[df_sizes < min_length].sort_values(ascending=False)
    len_to_concat = to_concat.shape[0]
    # This one will have the indices of the smallest big enough dataframes,
    # ordered by ascending length.
    concat_to = (df_sizes.loc[df_sizes >= min_length]
                         .sort_values()
                         .iloc[:len_to_concat])

    for i in range(len_to_concat):
        # We concatenate the biggest too small with the smallest big enough, and
        # then go down the list.
        idx_big_df = concat_to.index.values[i]
        idx_small_df = to_concat.index.values[i]
        big_df = results[idx_big_df]
        small_df = results[idx_small_df]
        results[idx_big_df] = pd.concat([big_df, small_df],
                                        ignore_index=True, sort=False)
    nr_dfs = len(results)
    all_indices = np.array(range(nr_dfs))
    # We get a boolean array stating whether each dataframe wasn't to be
    # concatenated because it was too short, so stating whether we want to
    # keep it in the list.
    keep_mask = np.isin(all_indices, to_concat.index.values, invert=True)
    results = [results[i] for i in range(nr_dfs) if keep_mask[i]]
    return results


def process(tweets_loc_df, places_geodf, langs_agg_dict,
            text_col='text', min_nr_words=4, cld='pycld2',
            latlon_proj='epsg:4326'):
    '''
    Takes a raw dataframe of tweets `tweets_loc_df`, filters it and assigns to
    each tweet a geometry, based on its coordinates if available, or through its
    `place_id` and the geometries of `places_geodf`. Then it assigns a language
    to every tweet when possible, and returns the whole dataframe.
    '''
    # Happened for Quebec to get empty dataframes, to avoid errors we return
    # here and get rid of those later on.
    if tweets_loc_df.shape[0] == 0:
        return None

    has_gps = tweets_loc_df['coordinates'].notnull()
    tweets_places_df = tweets_loc_df.loc[~has_gps].join(
        places_geodf[['geometry', 'area']], on='place_id', how='left')
    # The geometry of the tweets with GPS coordinates is the Point associated
    # to them.
    tweets_loc_df.loc[has_gps, 'geometry'] = (
        tweets_loc_df.loc[has_gps, 'coordinates']
                     .apply(lambda x: Point(x['coordinates'])))
    tweets_loc_df = geopd.GeoDataFrame(tweets_loc_df, crs=latlon_proj)
    tweets_loc_df.loc[has_gps, 'geometry'] = (
        tweets_loc_df.loc[has_gps, 'geometry'].to_crs(places_geodf.crs))
    # Since the projection was done on part of the GeoSeries, the crs parameter
    # doesn't get changed, so we do it manually.
    tweets_loc_df.crs = places_geodf.crs
    # We assign the area of points to 0, and at the same time initialize the
    # whole column, whose values will change for tweets without GPS coordinates.
    tweets_loc_df['area'] = 0
    # We add the geometry of the place to the tweets without GPS coordinates
    tweets_loc_df.loc[~has_gps, 'geometry'] = tweets_places_df['geometry']
    tweets_loc_df.loc[~has_gps, 'area'] = tweets_places_df['area']
    tweets_loc_df = (tweets_loc_df.rename(columns={'lang': 'twitter_lang'})
                                  .drop(columns=['coordinates']))
    print('starting lang detect')
    tweets_lang_df = text_process.lang_detect(tweets_loc_df, text_col=text_col,
        min_nr_words=min_nr_words, cld=cld, langs_agg_dict=langs_agg_dict)
    print('chunk lang detect done')
    return tweets_lang_df


def prep_resid_attr(tweets_lang_df, cells_in_area_df, max_place_area,
                    cc_timezone):
    '''
    Prepares the residence attribution of users whose tweets are within
    `tweets_lang_df`, by assigning each tweet either to a cell contained in
    `cells_in_area_df` if we have its precise geolocation, or to a place. Also
    flags whether a tweet was made within or outside work hours, considering
    the appropriate timezone `cc_timezone`.
    '''
    relevant_area_mask = tweets_lang_df['area'] < max_place_area
    tweets_df = tweets_lang_df.loc[relevant_area_mask].copy()
    tweets_df = isin_workhour_det(tweets_df, cc_timezone)
    has_gps = tweets_df['area'] == 0
    tweets_cells_df = geopd.sjoin(
        tweets_df.loc[has_gps], cells_in_area_df,
        op='within', rsuffix='cell', how='inner')
    tweets_places_df = tweets_df.loc[~has_gps]
    print('chunk preparation of residence attribution done')
    return tweets_cells_df, tweets_places_df


def isin_workhour_det(tweets_df, cc_timezone):
    tweets_df['hour'] = (tweets_df['created_at'].dt.tz_localize('UTC')
                                                .dt.tz_convert(cc_timezone)
                                                .dt.hour)
    # Tweets are considered in work hours if they were made between 8 and 18
    # outside of the week-end (weekday goes from 0 (Monday) to 6 (Sunday)).
    tweets_df['isin_workhour'] = (
        (tweets_df['hour'] > 7)
        & (tweets_df['hour'] < 18)
        & (tweets_df['created_at'].dt.weekday < 5))
    return tweets_df
