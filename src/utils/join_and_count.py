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
        init_index = pd.MultiIndex(levels=init_values, codes=init_values,
                                   names=groupby_cols)
        total_counts = pd.DataFrame([], index=init_index, columns=[count_col])
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
