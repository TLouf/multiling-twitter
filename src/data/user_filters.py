import pandas as pd

def get_months_activity(tweets_df, ref_year=2015, id_col='id', uid_col='uid',
                        dt_col='created_at'):
    '''
    From 'tweets_df', a DataFrame of tweets, with tweets and users uniquely
    identified by values in columns 'id_col' and 'uid_col', returns a DataFrame
    with the months in which the users have tweeted, order by user and month
    number.
    '''
    nr_users = len(tweets_df[uid_col].unique())
    print('There are {} distinct users in the dataset.'.format(nr_users))
    tweeted_months_users = tweets_df.loc[:, [id_col, uid_col]]
    # We get an integer representing the month number relative to the first
    # month of 'ref_year'
    month_series = tweets_df[dt_col].dt.month
    year_series = tweets_df[dt_col].dt.year - ref_year
    tweeted_months_users['month'] = year_series*12 + month_series
    tweeted_months_users.drop(columns=[id_col], inplace=True)
    tweeted_months_users.sort_values(by=[uid_col, 'month'], inplace=True)
    # We only keep one line per month:
    tweeted_months_users = tweeted_months_users.drop_duplicates()
    return tweeted_months_users


def consec_months(agg_tweeted_months_users, nr_consec_months=3, uid_col='uid'):
    '''
    From 'agg_tweeted_months_users', a DataFrame of all the months in which the
    users have tweeted, obtained by scanning through the whole dataset and
    applying 'get_months_activity' to it (preferably in chunks), returns a
    Series of all the IDs of users considered to be locals. Is considered local
    a user who has tweeted wihin at least three consecutive months.
    '''
    # We count the number of months in which a user has tweeted in total
    count_months_tweeted = (agg_tweeted_months_users.groupby(uid_col)['month']
                                                    .transform('size'))
    # If it's less than 'nr_consec_months', we drop them
    agg_tweeted_months_users = agg_tweeted_months_users.loc[
        count_months_tweeted >= nr_consec_months]
    nr_users = len(agg_tweeted_months_users[uid_col].unique())
    print('There are {} distinct users left in the dataset.'.format(nr_users))
    # At first the following was implmented on a user basis (using groupby), but
    # in the end it's cheaper to shift everything first, and then check we're on
    # the same uid. So first, every row is shifted downwards by two rows (the
    # first two are then NaN).
    shifted_df = agg_tweeted_months_users.shift(nr_consec_months-1)
    # If we had nr_consec_months consecutive months of activity, then we
    # consider them as local
    mask_consec_months = (agg_tweeted_months_users['month']
                          - shifted_df['month']) == nr_consec_months - 1
    # But here we shifted into other uids, se we should also check it's the same
    # uid which had the consecutive months of activity
    mask_same_uid = agg_tweeted_months_users[uid_col] == shifted_df[uid_col]
    # Here we lost the uid in the process, and kept only the same index as
    # agg_tweeted_months_users, so we retrieve the uid from a join with it,
    filter_df = agg_tweeted_months_users.join(
        (mask_same_uid & mask_consec_months).rename('filter'))
    # and then if any row is the result of three consecutive months of activity,
    # we mark the user as local
    is_uid_local = filter_df.groupby(uid_col)['filter'].any()
    local_uid_series = is_uid_local.loc[is_uid_local]
    nr_users = len(local_uid_series)
    print('There are {} distinct users left in the dataset.'.format(nr_users))
    return local_uid_series
