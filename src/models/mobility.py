import numpy as np
import pandas as pd
import geopandas as geopd
from ipfn import ipfn

def muni_to_cell(census_mobility, muni_in_cells):
    '''
    From the inter-municipality mobility, calculates the inter-cell mobility,
    using the ratio of overlap of each municipality with cells, which is given
    in `muni_in_cells`.
    '''
    cell_mobility = census_mobility.join(
        muni_in_cells.rename_axis(['from_cell_id', 'from_muni_id']))
    dest_muni_in_cells = (
        muni_in_cells.rename(columns={'ratio': 'to_ratio'})
                     .rename_axis(['to_cell_id', 'to_muni_id']))
    cell_mobility = cell_mobility.join(dest_muni_in_cells)
    cell_mobility['whole_ratio'] = (cell_mobility['ratio']
                                    * cell_mobility['to_ratio'])
    cell_mobility['cell_count'] = (cell_mobility['whole_ratio']
                                   * cell_mobility['count'])
    # Have to get rid of NaN before converting to int, which can happen for
    # exclaves like Llivia
    cell_mobility = cell_mobility.loc[cell_mobility['cell_count'] >= 1]
    cell_mobility['cell_count'] = cell_mobility['cell_count'].astype(int)
    return cell_mobility


def rescale_data(cell_plot_df, census_df, plot_lings_dict,
                 min_local_count=5):
    '''
    Upscaling of lang counts according to whole commuting population count from
    `census_df`, and to real proportion of each lang (given in
    `plot_lings_dict`). Uses Iterative Proportional Fitting (IPF) which allows
    to make marginal sums match.  We exclude cells where Twitter data isn't
    significant enough with a min local count, have to check that this doesn't
    exclude too many people which are present in the census (compare original
    total population to the one of rescaled_df).
    '''
    cell_plot_df.index = cell_plot_df['cell_id']
    is_enough = cell_plot_df['local_count'] > min_local_count
    rescaled_df = cell_plot_df.loc[is_enough].copy()
    has_twitter_data = census_df.index.isin(rescaled_df.index)

    census_cell_counts = census_df.loc[has_twitter_data, 'local_count']
    census_ling_counts = pd.Series(
        {ling_dict['count_col']: ling_dict['ratio']
         for ling_dict in plot_lings_dict.values()},
        name='total')
    census_ling_counts *= census_cell_counts.sum()

    count_ling_cols = [ling_dict['count_col']
                       for ling_dict in plot_lings_dict.values()]
    ling_mat = rescaled_df[count_ling_cols].to_numpy()
    aggregates = [census_ling_counts.values,
                  census_cell_counts.values]
    dimensions = [[1], [0]]
    IPF = ipfn.ipfn(ling_mat, aggregates, dimensions)
    # The algorithm changes the original ling_mat (so careful!)
    ling_mat = IPF.iteration()
    rescaled_df = geopd.GeoDataFrame(
        ling_mat, index=rescaled_df.index, columns=count_ling_cols,
        geometry=rescaled_df.geometry, crs=rescaled_df.crs)
    rescaled_df['local_count'] = sum([rescaled_df[col]
                                      for col in count_ling_cols])
    rescaled_df['total_count'] = rescaled_df['local_count']
    rescaled_df['cell_id'] = rescaled_df.index
    return rescaled_df


def commut_by_grp(cell_mobility, cell_plot_df, plot_lings_dict):
    '''
    Distributes the global commuting given by `cell_mobility` over all lingual
    groups, assuming that commuting is L-group independent. Presupposes that
    cell_plot_df has been rescaled to real ling ratios through `rescale_data`.
    '''
    lings = list(plot_lings_dict)
    cols = ['from_cell_id', 'to_cell_id']
    cell_mobility = cell_mobility.groupby(cols)[['cell_count']].sum()
    nr_commuters = cell_mobility['cell_count'].sum()
    for ling in lings:
        cell_plot_df['prop_'+ling] = (cell_plot_df['count_'+ling]
                                      / cell_plot_df['local_count'])

    count_prop_cols = (['count_'+ling for ling in lings]
                       + ['prop_'+ling for ling in lings])
    cell_mobility = cell_mobility.join(cell_plot_df[count_prop_cols],
                                       on='from_cell_id', how='inner')
    cell_mobility = cell_mobility.loc[(slice(None), cell_plot_df.index), :]
    new_nr_commuters = cell_mobility['cell_count'].sum()
    print(f'We lost {nr_commuters - new_nr_commuters} commuters out of '
          f'{nr_commuters} because of holes in cell_plot_df')
    # Each commuter is assigned to a L-group randomly, according to the
    # proportion of each group in the cell (which sum to 1 and define a proper
    # distirbution).
    rand_draw = [np.random.random(size) 
                 for size in cell_mobility['cell_count'].values]
    # ling_A corresponds to lings[0], and ling_B to lings[1]
    draws_vs_props = zip(rand_draw,
                         cell_mobility['prop_'+lings[0]].values,
                         cell_mobility['prop_'+lings[1]].values)
    commut_mono = zip(*[
        (np.sum(proba_arr < ling_A_th), np.sum(proba_arr > 1 - ling_B_th))
        for proba_arr, ling_A_th, ling_B_th in draws_vs_props])
    (cell_mobility['commut_'+lings[0]],
     cell_mobility['commut_'+lings[1]]) = commut_mono
    # The bilinguals are assigned the rest
    cell_mobility['commut_'+lings[2]] = (cell_mobility['cell_count']
                                         - cell_mobility['commut_'+lings[0]]
                                         - cell_mobility['commut_'+lings[1]])
    return cell_mobility


def get_user_dict(cell_mobility, plot_lings_dict):
    '''
    From the multiindex dataframe containing the mobility data, giving how many
    commute from one cell to another for all pairs where there's commuting,
    creates a dictionary containing the list of lings, resident cell and work
    cell, thus creating a synthetic population of agents. Presupposes that
    cell_mobility has been run through `commut_by_grp`.
    '''
    user_dict = {'ling': [], 'res_cell_id': [], 'work_cell_id': []}
    for (from_cell_id, to_cell_id), cell_data in cell_mobility.iterrows():
        cell_count = int(cell_data['cell_count'])
        user_dict['res_cell_id'].extend([from_cell_id] * cell_count)
        user_dict['work_cell_id'].extend([to_cell_id] * cell_count)
        for ling in plot_lings_dict:
            user_dict['ling'].extend([ling] * int(cell_data['commut_'+ling]))

    return user_dict


def df_to_nu(cell_mobility, tau=2):
    '''
    From the multiindex dataframe containing the mobility data, giving how many
    commute from one cell to another for all pairs where there's commuting,
    calculates the corresponding matrix nu (see SI).
    '''
    N = cell_mobility.groupby('from_cell_id')['cell_count'].sum()
    df_to_mat_idx = {idx: i for i, idx in enumerate(N.index.values)}
    N = N.values
    sigma = np.zeros((N.shape[0], N.shape[0]))
    for (from_cell_id, to_cell_id), commuters in cell_mobility['cell_count'].items():
        i_origin = df_to_mat_idx.get(from_cell_id)
        i_dest = df_to_mat_idx.get(to_cell_id)
        # To account for cells where only incoming commuters but no residents,
        # plus we put 0 on the diagonal, as sigma is only outgoing
        # (see Sattenspiel)
        if (i_origin is not None) and (i_dest is not None) and (i_origin != i_dest):
            sigma[i_origin, i_dest] += commuters
    # To convert this to a ratio, we simply divide by the population of the cell
    # of residence. This is already a rate, because the data we have is the
    # number of persons commuting per day. So the unit is already 1/day.
    # At this point sigma_ij is proportion of i going to j
    sigma = (sigma.T / N).T
    nu = sigma + tau * np.eye(sigma.shape[0])
    sigma_sum = np.sum(sigma, axis=1)
    nu = (nu.T / (tau + sigma_sum)).T
    return nu, N, df_to_mat_idx
