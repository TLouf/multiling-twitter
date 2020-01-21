import pandas as pd
import geopandas as geopd
import json
from shapely.geometry import Point

def increment_counts(total_counts, data_to_count, groupby_cols,
                     count_col='count'):
    '''
    Increments 'total_counts' by the counts by 'groupby_cols' (a list of str)
    in 'data_to_count'.
    '''
    if total_counts is None:
        init_values = [[] for x in range(len(groupby_cols))]
        init_index = pd.MultiIndex(
            levels=init_values, codes=init_values, names=groupby_cols)
        total_counts = pd.DataFrame(
            [], index=init_index, columns=[count_col])
    new_counts = (data_to_count.assign(**{count_col: 0})
                               .groupby(groupby_cols)[count_col].count())
    total_counts = increment_join(total_counts, new_counts, count_col=count_col)
    return total_counts


def increment_join(counts_df, counts_series, count_col='count'):
    '''
    Joins counts_series to counts_df, and increments count_col with its values.
    Also works on frames and series with a multi index.
    '''
    two_counts_df = counts_df.join(counts_series, how='outer', rsuffix='_new')
    new_count_col = count_col + '_new'
    two_counts_df[[count_col, new_count_col]] = (
        two_counts_df[[count_col, new_count_col]].fillna(value=0))
    two_counts_df[count_col] += two_counts_df[new_count_col]

    # If counts_df is just an initialized empty df, its dtype will be the
    # default, 'object'. So the result is to be casted to an integer if it's
    # empty or an integers Series, and if the counts_series was also integers.
    if ((counts_df[count_col].dtype == 'int' or len(counts_df) == 0)
            and counts_series.dtype == 'int'):
        two_counts_df[count_col] = two_counts_df[count_col].astype('int')

    two_counts_df = two_counts_df.drop(columns=[new_count_col])
    return two_counts_df


def get_counts(tweets_file_path, cells_df, dtype_dict=None,
        data_proj='epsg:4326', chunksize=10000):
    '''
    Iterates over a json file containing tweets and their geo-location, and
    counts how many were sent from each cell defined in 'cell_df'.
    '''
    cells_tweets_counts = pd.Series(name='count', dtype='int64')
    crs = {'init': data_proj}
    with open(tweets_file_path) as f:
        chunks = pd.read_json(f, lines=True, chunksize=chunksize,
                              dtype=dtype_dict)
        while True:
            try:
                # The following can return a ValueError when parsing json,
                # because of some problematic lines
                tweets_df = next(chunks)
                geometry = tweets_df['coordinates'].apply(lambda x: Point(x))
                tweets_gdf = geopd.GeoDataFrame(
                    tweets_df, crs=crs, geometry=geometry)
                tweets_within_cells = geopd.sjoin(tweets_gdf, cells_df,
                                                  op='within', rsuffix='cell')
                cells_tweets_counts = increment_counts(
                    cells_tweets_counts, tweets_within_cells, 'index_cell')


            except ValueError:
                # to skip problematic lines
                print('ValueError encountered, {} tweets were skipped'.format(
                    chunksize))
            except StopIteration:
                # when we read all the chunks
                print('done')
                break

    return cells_tweets_counts


def get_counts_primitive(tweets_file_path, cells_df, tweet_cols,
                         data_proj='epsg:4326'):
    '''
    Has the same purpose as 'get_counts', except it reads the file line by line
    using the json package
    '''
    nr_errors = 0
    crs = {'init': data_proj}
    cells_tweets_counts = pd.Series(name='count', dtype='int64')
    tweets_list = []
    with open(tweets_file_path) as f:
        for i, line in enumerate(f):
            # Some lines in the file don't seem to be formatted correctly,
            # so we need to pass those when they raise an error.
            try:
                tweet_dict = json.loads(line, strict=False)
                tweets_list.append(tweet_dict.values())

                if i%1e3 == 0:
                    tweets_df = pd.DataFrame(tweets_list, columns=tweet_cols)
                    x = tweets_df['coordinates'].apply(lambda x: x[0]).values
                    y = tweets_df['coordinates'].apply(lambda x: x[1]).values
                    geometry = pd.Series(map(Point, zip(x,y)))
                    tweets_gdf = geopd.GeoDataFrame(
                        tweets_df, crs=crs, geometry=geometry)
                    tweets_within_cells = geopd.sjoin(
                        tweets_gdf, cells_df, op='within', rsuffix='cell')
                    cells_tweets_counts = increment_counts(
                        cells_tweets_counts, tweets_within_cells, 'index_cell')
                    tweets_list = []

            except json.decoder.JSONDecodeError:
                nr_errors += 1
    print('{} tweets were skipped because of decoding errors'.format(nr_errors))
    return cells_tweets_counts
