import pandas as pd
import geopandas as geopd
import json
from shapely.geometry import Point

def increment_counts(cells_tweets_counts, tweets_gdf, cells_df):
    '''
    Increments the counts of tweets sent from each cell in 'cells_tweets_counts'
    with the ones in 'tweets_gdf' if they were sent from a cell of 'cell_df'.
    '''
    tweets_within_cells = geopd.sjoin(tweets_gdf, cells_df, op='within', rsuffix='cell')
    new_cells_tweets_counts = tweets_within_cells.groupby(['index_cell']).size()
    new_cells_tweets_counts.rename('count_new', inplace=True)
    new_cells_tweets_counts = new_cells_tweets_counts.to_frame()
    two_counts_df = new_cells_tweets_counts.join(cells_tweets_counts, how='outer')
    two_counts_df.fillna(value=0, inplace=True)
    cells_tweets_counts = two_counts_df['count'] + two_counts_df['count_new']
    cells_tweets_counts = cells_tweets_counts.rename('count')
    return cells_tweets_counts


def get_counts(tweets_file_path, cells_df, dtype_dict=None,
        data_proj='epsg:4326', chunksize=10000):
    '''
    Iterates over a json file containing tweets and their geo-location, and
    counts how many were sent from each cell defined in 'cell_df'.
    '''
    cells_tweets_counts = pd.Series(name='count', dtype='int64')
    crs = {'init': data_proj}
    with open(tweets_file_path) as f:
        chunks = pd.read_json(f, lines=True, chunksize=chunksize, dtype=dtype_dict)
        while True:
            try:
                # The following can return a ValueError when parsing json,
                # because of some problematic lines
                tweets_df = next(chunks)
                geometry = tweets_df['coordinates'].apply(lambda x: Point(x))
                tweets_gdf = geopd.GeoDataFrame(
                    tweets_df, crs=crs, geometry=geometry)
                cells_tweets_counts = increment_counts(
                    cells_tweets_counts, tweets_gdf, cells_df)


            except ValueError:
                # to skip problematic lines
                print('ValueError encountered, {} tweets were skipped'.format(
                    chunksize))
            except StopIteration:
                # when we read all the chunks
                print('done')
                break

    return cells_tweets_counts


def get_counts_primitive(tweets_file_path, cells_df, tweet_cols, data_proj='epsg:4326'):
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
                ## OR:
#                 b = line.split('":')
#                 tweet_id = int(b[1].split(',')[0])
#                 uid = int(b[2].split(',')[0])
#                 created_at = b[3].split('"')[1]
#                 start_list = b[4][2:].split(',')
#                 x = float(start_list[0])
#                 y = float(start_list[1][1:-4])
# #                 coords = [x, y]
#                 tweets_list.append([tweet_id, uid, created_at, x, y])

                if i%1e3 == 0:
                    tweets_df = pd.DataFrame(tweets_list, columns=tweet_cols)
#                     x = tweets_df['x'].values
#                     y = tweets_df['y'].values
                    x = tweets_df['coordinates'].apply(lambda x: x[0]).values
                    y = tweets_df['coordinates'].apply(lambda x: x[1]).values
                    geometry = pd.Series(map(Point, zip(x, y)))
#                     geometry = tweets_df['coordinates'].apply(lambda x: Point(x))
                    tweets_gdf = geopd.GeoDataFrame(
                        tweets_df, crs=crs, geometry=geometry)
                    cells_tweets_counts = increment_counts(
                        cells_tweets_counts, tweets_gdf, cells_df)
                    tweets_list = []

            except json.decoder.JSONDecodeError:
                nr_errors += 1
    print('{} tweets were skipped because of decoding errors.'.format(nr_errors))
    return cells_tweets_counts
