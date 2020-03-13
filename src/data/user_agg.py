import pandas as pd
import geopandas as geopd
import src.data.access as data_access
import src.data.process as data_process
import src.visualization.helpers as helpers_viz

def users_months(raw_tweets_df, ref_year=2015):
    '''
    Counts the number of tweets each user posted each month.
    '''
    # We get an integer representing the month number relative to the first
    # month of `ref_year`.
    month_series = raw_tweets_df['created_at'].dt.month
    year_series = raw_tweets_df['created_at'].dt.year - ref_year
    raw_tweets_df['month'] = year_series*12 + month_series
    months_counts = (raw_tweets_df
                        .assign(**{'count': 0})
                        .groupby(['uid', 'month'])['count']
                        .count())
    nr_users = len(raw_tweets_df['uid'].unique())
    print(f'There are {nr_users} distinct users in this chunk.')
    return months_counts


def get_lang_loc_habits(df_access, get_df_fun, valid_uids, places_geodf,
                        langs_agg_dict, cells_df_list, max_place_area,
                        cc_timezone, text_col='text', min_nr_words=4,
                        cld='pycld2', latlon_proj='epsg:4326'):
    '''
    Given a dataframe access and `get_df_fun` (see
    `data_access.yield_tweets_access`), returns three Series aggregated at the
    user level, about users in `valid_uids`. The first has how many tweets each
    user has posted in each language they tweeted. The second and the third
    series contains the counts of their tweets in the places defined in
    `places_geodf` and in the cells defined in `cells_in_area_df`, divided
    between those posted within and outside work hours (see
    `data_process.prep_resid_attr` about the definition of work hours).
    '''
    tweets_df = data_process.process(
        df_access, get_df_fun, valid_uids, places_geodf, langs_agg_dict,
        min_nr_words=min_nr_words, cld=cld)

    groupby_cols = ['uid', 'cld_lang']
    user_langs_counts = (tweets_df
                            .assign(**{'count': 0})
                            .groupby(groupby_cols)['count'].count())

    relevant_area_mask = tweets_df['area'] < max_place_area
    tweets_df = tweets_df.loc[relevant_area_mask]
    tweets_df = data_process.isin_workhour_det(tweets_df, cc_timezone)
    has_gps = tweets_df['area'] == 0
    tweets_places_df = tweets_df.loc[~has_gps]
    groupby_cols = ['uid', 'place_id', 'isin_workhour']
    user_places_habits = (tweets_places_df
                            .assign(**{'count': 0})
                            .groupby(groupby_cols)['count'].count())

    groupby_cols = ['uid', 'cell_id', 'isin_workhour']
    user_cells_habits_list = []
    for cells_in_area_df in cells_df_list:
        tweets_cells_df = geopd.sjoin(tweets_df.loc[has_gps], cells_in_area_df,
            op='within', rsuffix='cell', how='inner')
        user_cells_habits = (tweets_cells_df
                                .assign(**{'count': 0})
                                .groupby(groupby_cols)['count'].count())
        user_cells_habits_list.append(user_cells_habits)

    return user_langs_counts, user_cells_habits_list, user_places_habits


def to_count_by_area(users_counts, users_area, output_col='count'):
    '''
    From users counts (by uid and possibly another level in the index, like
    a language) and the id of the area each resides in, makes the corresponding
    counts by area of residence.
    '''
    groupby_cols = users_area.name
    if type(users_counts.index) == pd.MultiIndex:
        # `users_counts` must have 'uid' as first level and a language group
        # as second level
        groupby_cols = [users_counts.index.names[1], groupby_cols]
    agg_count = (users_counts.join(users_area, how='inner')
                             .groupby(groupby_cols)
                             .size()
                             .rename(output_col))
    return agg_count


def get_lang_grp(user_langs_counts, area_dict, lang_relevant_prop=0.1,
                 lang_relevant_count=5, fig_dir=None, show_fig=False):
    '''
    Out of the counts of tweets by language of each user in `user_langs_counts`,
    determines which one are relevant for each user, before assigning them their
    spoken languages in `user_langs_agg`.
    '''
    total_per_user = (user_langs_counts.groupby('uid')['count']
                                       .sum()
                                       .rename('user_count'))
    user_langs_agg = user_langs_counts.join(total_per_user).assign(
        prop_lang=lambda df: df['count'] / df['user_count'])
    prop_relevant = user_langs_agg['prop_lang'] > lang_relevant_prop
    count_relevant = user_langs_agg['count'] > lang_relevant_count
    user_langs_agg = user_langs_agg.loc[prop_relevant | count_relevant]
    uid_with_lang = user_langs_agg.index.levels[0].values
    print(f'We were able to attribute at least one language to '
          f'{len(uid_with_lang)} users')
    if fig_dir:
        helpers_viz.top_lang_speakers(
            user_langs_agg, area_dict, lang_relevant_count, lang_relevant_prop,
            fig_dir=fig_dir, show=show_fig)
    return user_langs_agg


def get_ling_grp(user_langs_agg, area_dict, lang_relevant_prop=0.1, 
                 lang_relevant_count=5, fig_dir=None, show_fig=False):
    '''
    Out of the spoken languages of each user in `user_langs_agg`, assigns them
    to a local lingual group: that is out of the languages spoken in the area,
    are they monolinguals of a language or bi-tri-quadrilinguals?
    '''
    local_langs = area_dict['local_langs']
    users_ling_grp = user_langs_agg.reset_index(level='cld_lang')
    # `user_langs_agg` is sorted by user and language, because of the groupby in
    # `increment_counts`. Thus, when we concatenate the languages with sum()
    # here, they're already sorted so we won't get both 'frit' and 'itfr' for
    # instance.
    local_mask = users_ling_grp['cld_lang'].isin(local_langs)
    users_ling_grp = (users_ling_grp.loc[local_mask, 'cld_lang']
                                    .groupby('uid')
                                    .apply(lambda langs: 'ling_'+langs.sum())
                                    .rename('ling_grp')
                                    .to_frame()
                                    .groupby(['uid', 'ling_grp'])
                                    .first())
    ling_counts = (users_ling_grp.reset_index()
                                 .groupby('ling_grp')
                                 .size()
                                 .sort_values(ascending=False))
    multiling_grps = ling_counts.index.values
    total_count = total_count = len(user_langs_agg.index.levels[0])
    if fig_dir:
        helpers_viz.ling_grps(multiling_grps, ling_counts, total_count,
                              area_dict, lang_relevant_count,
                              lang_relevant_prop, fig_dir=fig_dir, 
                              show=show_fig)
    return users_ling_grp, multiling_grps


def get_home_place(raw_user_habits, place_id_col='place_id', relevant_th=0.1):
    '''
    From the counts of tweets of users by place and time (in a categorical
    aggregate, whether the tweets were made within or outside work hours),
    attributes a place of residence to each user. The returned series has the
    user IDs for indices, and the IDs of their place of residence as values.
    '''
    user_habits = raw_user_habits.reset_index(level='isin_workhour')
    count_by_place = get_prop(user_habits, 'uid', place_id_col)
    # Better to have uid and place_id in index for the join here:
    user_habits = user_habits.join(count_by_place, how='left')
    # Let's select places where users tweeted more than relevant_th*100% of the
    # time within or outside work hours:
    user_habits = user_habits.loc[user_habits['prop'] > relevant_th]
    # Then in these places we take the one where they tweeted outside workhours
    # the most
    user_home_place = user_habits.reset_index()
    user_home_place = user_home_place.loc[~user_home_place['isin_workhour']]
    user_home_place = (user_home_place.sort_values(by=['uid', 'count'])
                                      .groupby('uid')[place_id_col]
                                      .last())
    # For all these people, we found their place of residence. Now let's deal
    # with users for which there's no place where they tweeted more than
    # relevant_th*100% of the time outside of workhours. We'll simply take the
    # place from which they tweeted the most.
    user_pref_place = (count_by_place.reset_index(level=place_id_col)
                                     .sort_values(by=['uid', 'place_count'])
                                     .groupby('uid')[place_id_col]
                                     .last())
    user_home_place = user_pref_place.to_frame().join(
        user_home_place, how='left', rsuffix='_home')
    home_found = user_home_place[place_id_col+'_home'].notnull()
    user_home_place.loc[home_found, place_id_col] = user_home_place.loc[
        home_found, place_id_col+'_home']
    user_home_place = user_home_place[place_id_col]
    return user_home_place


def get_residence(user_cells_habits, user_places_habits, place_relevant_th=0.1,
                  cell_relevant_th=0.1):
    '''
    Gets either the cell or, if not suited, the place of residence of every
    user, based on heir tweeting habits in `user_cells_habits` and 
    `user_places_habits`. These two series count the number of tweets of every
    user by cell/place by period of the day (inside or outside work hours).
    '''
    user_counts_in_cells = (user_cells_habits.groupby('uid')['count']
                                             .sum()
                                             .rename('user_count'))
    user_home_cell = user_cells_habits.join(user_counts_in_cells, how='inner')
    user_home_cell['prop'] = (user_home_cell['count']
                              / user_home_cell['user_count'])
    user_home_cell = (user_home_cell.loc[user_home_cell['prop']
                                         > cell_relevant_th]
                                    .xs(False, level='isin_workhour')
                                    .reset_index()
                                    .sort_values(by=['uid', 'count'])
                                    .groupby('uid')['cell_id']
                                    .last())

    users_with_cell = user_home_cell.index.values
    user_home_place = get_home_place(user_places_habits,
                                     place_id_col='place_id',
                                     relevant_th=place_relevant_th)
    user_only_place = user_home_place.reset_index()
    user_only_place = (
        user_only_place.loc[~user_only_place['uid'].isin(users_with_cell)]
                       .set_index('uid')
                       .loc[:, 'place_id'])
    return user_home_cell, user_only_place


def get_prop(df, first_lvl_id, second_lvl_id):
    '''
    From 'df', creates a new dataframe 'prop_df' with a two-level index built
    from the columns 'first_lvl_id' and 'second_lvl_id'. It contains two first
    columns with the counts in 'df' grouping by these two levels and grouping
    only by the 'first_lvl_id'. A third column contains the ratio between these
    two counts.
    '''
    prop_df = (df.groupby([first_lvl_id, second_lvl_id])['count']
                 .sum()
                 .to_frame())
    first_lvl_count = first_lvl_id[:-2] + 'count'
    second_lvl_count = second_lvl_id[:-2] + 'count'
    prop_df = prop_df.rename(
        columns={'count': second_lvl_count})
    prop_df[first_lvl_count] = (prop_df.groupby(first_lvl_id)
                                       .transform('sum'))
    prop_df['prop'] = prop_df[second_lvl_count] / prop_df[first_lvl_count]
    return prop_df
