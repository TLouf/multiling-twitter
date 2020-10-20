import numpy as np


def get_cell_ratio(users_contacts, user_home_cell):
    user_interactions = users_contacts.join(user_home_cell)
    dest_user_home_cell = (user_home_cell.copy()
                                         .rename(columns={'ratio': 'to_ratio'}))
    dest_user_home_cell.index = (
        dest_user_home_cell.index.rename(['to_uid', 'to_cell_id'],
                                         level=['uid', 'cell_id']))
    user_interactions = user_interactions.join(dest_user_home_cell)
    user_interactions['whole_ratio'] = (user_interactions['ratio']
                                        * user_interactions['to_ratio'])
    # Check it sums to one, except people for whom we didn't find a residence,
    # who have a null sum
    cols = ['uid', 'to_uid', 'cld_lang']
    interaction_sum = (user_interactions.groupby(cols)['whole_ratio']
                                        .sum())
    mask = (np.abs(interaction_sum-1) > 1e-10) & (interaction_sum > 0)
    if len(interaction_sum.loc[mask]) > 0:
        print('Warning: the ratios do not sum to 1')
    
    return user_interactions
