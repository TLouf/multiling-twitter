import pandas as pd

def get_residence(raw_user_habits, place_id_col='place_id', relevant_th=0.1):
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
    user_habits = user_habits.loc[user_habits['place_prop'] > relevant_th]
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


def get_prop(df, first_lvl_id, second_lvl_id):
    '''
    From 'df', creates a new dataframe 'prop_df' with a two-level index built
    from the columns 'first_lvl_id' and 'second_lvl_id', and the counts in 'df'
    grouping by these two levels, grouping only by the 'first_lvl_id', and the
    ratio between these two.
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
    prop_df[second_lvl_id[:-2]+'prop'] = (prop_df[second_lvl_count]
                                             / prop_df[first_lvl_count])
    return prop_df
