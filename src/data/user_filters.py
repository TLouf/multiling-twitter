'''
Every filter returns a series with as index uids, and as values a boolean,
which in every case says whether this uid shoud be kept. Thus so some functions
may return a series full of False if it's users to exclude, or full of True
if they're the only users to keep.
'''
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import geopandas as geopd
from shapely.geometry import Point
import src.utils.join_and_count as join_and_count
import src.data.access as data_access
import src.data.user_agg as uagg

LOGGER = logging.getLogger(__name__)

def inc_months_activity(tweeted_months_users_agg, tweets_df, ref_year=2015,
                        id_col='id', uid_col='uid', dt_col='created_at'):
    '''
    From 'tweets_df', a DataFrame of tweets, with tweets and users uniquely
    identified by values in columns 'id_col' and 'uid_col', returns a DataFrame
    with the months in which the users have tweeted, ordered by user and month
    number.
    '''
    tweeted_months_users = tweets_df.loc[:, [id_col, uid_col]]
    # We get an integer representing the month number relative to the first
    # month of 'ref_year'.
    month_series = tweets_df[dt_col].dt.month
    year_series = tweets_df[dt_col].dt.year - ref_year
    tweeted_months_users['month'] = year_series*12 + month_series
    tweeted_months_users_agg = join_and_count.increment_counts(
        tweeted_months_users_agg, tweeted_months_users, [uid_col, 'month'])
    return tweeted_months_users_agg


def consec_months(tweeted_months_users_agg, nr_consec_months=3, uid_col='uid'):
    '''
    From 'tweeted_months_users_agg', a DataFrame of all the months in which the
    users have tweeted, obtained by scanning through the whole dataset and
    applying 'get_months_activity' to it (preferably in chunks), returns a
    Series of all the IDs of users considered to be locals. Is considered local
    a user who has tweeted within at least three consecutive months.
    '''
    # We count the number of months in which a user has tweeted in total.
    count_months_tweeted = (tweeted_months_users_agg.groupby(uid_col)
                                                    .transform('size'))
    # If it's less than 'nr_consec_months', we drop them.
    tweeted_months_users = tweeted_months_users_agg.loc[
        count_months_tweeted >= nr_consec_months]
    tweeted_months_users = tweeted_months_users.reset_index()
    nr_users = len(tweeted_months_users[uid_col].unique())
    LOGGER.info(f'There are {nr_users} users with at least {nr_consec_months} '
                'months of activity in the dataset.')
    # At first the following was implemented on a user basis (using groupby),
    # but in the end it's cheaper to shift everything first, and then check
    # we're on the same uid. So first, every row is shifted downwards by
    # nr_consec_months-1 rows (the first nr_consec_months-1 are then NaN).
    shifted_df = tweeted_months_users.shift(nr_consec_months-1)
    # If we had nr_consec_months consecutive months of activity, then we
    # consider them as local
    mask_consec_months = (tweeted_months_users['month']
                          - shifted_df['month']) == nr_consec_months - 1
    # But here we shifted into other uids, se we should also check it's the same
    # uid which had the consecutive months of activity
    mask_same_uid = tweeted_months_users[uid_col] == shifted_df[uid_col]
    local_uids = tweeted_months_users.loc[
        mask_same_uid & mask_consec_months, uid_col].unique()
    nr_users = len(local_uids)
    LOGGER.info(f'There are {nr_users} users considered local in the dataset, '
                f'as they have been active for {nr_consec_months} consecutive '
                'months in this area at least once.')
    local_uids = pd.Series(True, index=local_uids, name='local')
    local_uids.index.name = 'uid'
    return local_uids


def bot_activity(tweeted_months_users_agg, max_tweets_per_min=3):
    '''
    From 'tweeted_months_users_agg', a DataFrame of all the months in which the
    users have tweeted, obtained by scanning through the whole dataset and
    applying 'get_months_activity' to it (preferably in chunks), returns a
    Series of all the IDs of users considered to be bots. Is considered a bot
    a user who has tweeted more than 'max_tweets_per_min' per minute on average,
    taken over a period spanning from its first tweet to its last (in our
    dataset).
    '''
    tweeted_months_users = tweeted_months_users_agg.reset_index()
    grouped_by_uid = tweeted_months_users.groupby('uid')
    total_count = grouped_by_uid['count'].sum()
    months_extent = (1 + grouped_by_uid['month'].last()
                     - grouped_by_uid['month'].first())
    # Only 10% of tweets are extracted, so we need to be less lenient for this
    # filter by multiplying the count by 10.
    not_bot_mask = ((10 * total_count / (months_extent*30*24*60))
                    < max_tweets_per_min)
    # We only keep in the index the UIDs to exclude.
    bot_uids = not_bot_mask.rename('bot').loc[~not_bot_mask]
    bot_uids.index.name = 'uid'
    LOGGER.info(f'{len(bot_uids)} users have been found to be bots because of '
                f'their excessive activity, tweeting more than '
                f'{max_tweets_per_min} times per minute.')
    return bot_uids


def too_fast(raw_tweets_df, places_in_xy, max_distance, speed_th=280,
             point_proj='epsg:4326'):
    '''
    'places_in_xy' is the GeoDataFrame of the places with the geometry projected
    to x,y coordinates. The conversion is done outside the function for later
    re use.
    Speed filter: we'll take the distance function of Shapely which takes the
    minimum distance between two objects. Indeed, the naive method of taking
    the two centroids of the places and calculating the distance can cause
    numerous false positives. Indeed, if someone tags themself in a very large
    place, and then a very small one within or close to it, the two centroids
    can be very far apart, and the user actually probably hasn't moved much.
    Every physical variable is in SI units: distance in meters and time in
    seconds.
    '''
    tweets_df = raw_tweets_df.copy()
    has_gps = tweets_df['coordinates'].notnull()
    tweets_places_df = tweets_df.loc[~has_gps].join(
        places_in_xy, on='place_id', how='left')
    # The geometry of the tweets with GPS coordinates is the Point associated to
    # them.
    tweets_df.loc[has_gps, 'geometry'] = (
        tweets_df.loc[has_gps,'coordinates']
                 .apply(lambda x: Point(x['coordinates'])))
    tweets_df = geopd.GeoDataFrame(tweets_df, crs=point_proj)
    # We have to project these points to x,y coordinates to match the projection
    # of places.
    tweets_df.loc[has_gps, 'geometry'] = (
        tweets_df.loc[has_gps, 'geometry'].to_crs(places_in_xy.crs))
    # Since the projection was done on part of the GeoSeries, the crs parameter
    # doesn't get changed, so we do it manually.
    tweets_df.crs = places_in_xy.crs
    # We add the geometry of the place to the tweets without GPS coordinates
    tweets_df.loc[~has_gps, 'geometry'] = tweets_places_df['geometry']

    tweets_df = tweets_df.sort_values(by=['uid', 'created_at'])
    prev_tweets_df = tweets_df.shift(1)
    same_uid_mask = tweets_df['uid'] == prev_tweets_df['uid']
    tweets_df = tweets_df.loc[same_uid_mask]
    prev_tweets_df = prev_tweets_df.loc[same_uid_mask]
    tweets_df['delta_t'] = (tweets_df['created_at']
                            - prev_tweets_df['created_at']).dt.total_seconds()
    # delta_t can be too large because speed_th*delta_t > max distance in the
    # country, so in order to calculate as few distances between polygons as
    # possible, we filter out tweets_df where the delta_t is too large
    short_enough = speed_th*tweets_df['delta_t'] < max_distance
    tweets_df = tweets_df.loc[short_enough]
    prev_tweets_df = prev_tweets_df.loc[short_enough]
    # This line below is the expensive operation: it calculates the minimum
    # distance between two Polygons, as Shapely implemented it. This is the
    # minimum distance between them, so it is null when the polygons
    # intersect, which is what we want.
    tweets_df['delta_x'] = tweets_df['geometry'].distance(
        prev_tweets_df['geometry'])
    tweets_df = tweets_df.loc[tweets_df['delta_t'] > 0]
    tweets_df['speed'] = tweets_df['delta_x'] / tweets_df['delta_t']
    tweets_df = tweets_df.loc[tweets_df['speed'] > speed_th]
    too_fast_uids = pd.Series(
        False, index=tweets_df['uid'].unique(), name='too_fast')
    too_fast_uids.index.name = 'uid'
    LOGGER.info(f'{len(too_fast_uids)} users have been found in this chunk with'
                f' a speed exceeding {speed_th*3.6:n} km/h.')
    return too_fast_uids


def filters_chunk_pass(df_access, get_df_fun, areas_dict,
                       cols=None, ref_year=2015):
    return_dict = {}
    tweets_df = get_df_fun(df_access)
    for region, region_dict in areas_dict['regions'].items():
        places_geodf = region_dict['places_geodf']
        shape_df = region_dict['shape_df']
        area_bounds = shape_df.geometry.iloc[0].bounds
        # Get an upper limit of the distance that can be travelled inside the
        # area
        max_distance = np.sqrt((area_bounds[0]-area_bounds[2])**2 
                            + (area_bounds[1]-area_bounds[3])**2)
        region_tweets_df = data_access.filter_df(
            tweets_df, cols=cols, dfs_to_join=[places_geodf])
        months_counts = uagg.users_months(region_tweets_df, ref_year=ref_year)
        too_fast_uids = too_fast(region_tweets_df, places_geodf, max_distance)
        return_dict[region] = {}
        return_dict[region]['months_counts'] = months_counts
        return_dict[region]['too_fast_uids'] = too_fast_uids
    return return_dict
    
    
def get_valid_uids(areas_dict, get_df_fun, collect_filters_pass_res,
                   filters_pass_res, tweets_files_paths, cpus=8):
    cols = ['uid', 'created_at', 'place_id', 'coordinates']
    pool = mp.Pool(cpus)
    for i, df_access in enumerate(
            data_access.yield_tweets_access(tweets_files_paths)):
        LOGGER.info(f'- starting on chunk {i}')
        args = (df_access, get_df_fun, areas_dict)
        kwargs = {'cols': cols}
        pool.apply_async(
            filters_chunk_pass, args, kwargs,
            callback=collect_filters_pass_res, error_callback=print)
    pool.close()
    pool.join()
    
    for region, region_dict in areas_dict['regions'].items():
        region_name = region_dict['readable']
        tweeted_months_users = join_and_count.init_counts(['uid', 'month'])
        all_too_fast_uids = pd.Series([])
        all_too_fast_uids.index.name = 'uid'
        for res in filters_pass_res:
            months_counts = res[region]['months_counts']
            tweeted_months_users = join_and_count.increment_join(
                tweeted_months_users, months_counts)
            too_fast_uids = res[region]['too_fast_uids']
            all_too_fast_uids = (all_too_fast_uids * too_fast_uids).fillna(False)
                
        tweeted_months_users = tweeted_months_users['count']
        total_nr_users = len(
            tweeted_months_users.index.get_level_values('uid').unique())
        LOGGER.info(f'There are {total_nr_users} distinct users in the whole '
                    f'dataset in {region_name}.')
        
        local_uids = consec_months(tweeted_months_users)
        bot_uids = bot_activity(tweeted_months_users)
        # We have local_uids: index of uids with a column full of True, and
        # bot_uids: index of uids with a column full of False. When we multiply
        # them, the uids in local_uids which are not in bot_uids are assigned NaN,
        # and the ones which are in bot_uids are assigned False. When we convert to
        # the boolean type, the NaNs turn to True.
        valid_uids = (local_uids * bot_uids).astype('bool').rename('valid')
        valid_uids = valid_uids.loc[valid_uids]
        LOGGER.info(f'There are {len(all_too_fast_uids)} too fast users to '
                    f'filter out in the whole dataset in {region_name}.')
        valid_uids = ((valid_uids * all_too_fast_uids).astype('bool')
                                                    .rename('valid'))
        valid_uids = valid_uids.loc[valid_uids]
        LOGGER.info(f'This leaves us with {len(valid_uids)} valid users in the '
                    f'whole dataset in {region_name}.')
        areas_dict['regions'][region]['valid_uids'] = valid_uids
        
    return areas_dict
    