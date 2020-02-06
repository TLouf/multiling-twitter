import numpy as np
import pandas as pd
import src.visualization.helpers as helpers_viz
from sklearn.cluster import KMeans

def kl_div(obs, pred):
    '''
    Computes -obs*log(pred/obs), the elements over which one sums to get the
    Kullback Leibler divergence of between the distribution defined by all
    the obs and the one defined by all the pred. pred == 0 must imply obs == 0.
    '''
    # We add these two steps to avoid a 0*log(0), as in Python 0**0 = 1, but
    # 0*log(0) is not a number.
    den = obs**obs
    num = pred**obs
    return -np.log(num / den)


def max_kl(cell_plot_df, grp_count_col):
    '''
    Computes the maximum KL divergence value that a given group can attain in
    a cell.
    '''
    N = cell_plot_df['total_count'].sum()
    N_grp = cell_plot_df[grp_count_col].sum()
    return -np.log(N_grp / N)


def entropy(p):
    '''
    Computes -p*log(p), the elements over which one sums to get Shannon's
    entropy of a system defined by the probability distribution {p}.
    '''
    # Written this way to have 0*log 0 = 1.
    return -np.log(p**p)


def max_H(cell_plot_df, grp_count_col):
    return entropy(cell_plot_df['total_conc'].values).sum()


def calc_by_cell(cell_plot_df, grps_dict):
    '''
    Adds columns to cell_plot_df with multiple metrics of interest, for every
    group described in grps_dict.
    '''
    total_nr_users = cell_plot_df['total_count'].sum()
    cell_plot_df['total_conc'] = cell_plot_df['total_count'] / total_nr_users

    for grp, grp_dict in grps_dict.items():
        count_col = grp_dict['count_col']
        conc_col = f'conc_{grp}'
        prop_col = f'prop_{grp}'
        repr_col = f'repr_{grp}'
        H_col = f'H_{grp}'
        KL_col = f'KL_{grp}'

        grp_total = cell_plot_df[count_col].sum()
        cell_plot_df[conc_col] = cell_plot_df[count_col] / grp_total
        cell_plot_df[prop_col] = (cell_plot_df[count_col]
                                  / cell_plot_df['total_count'])
        cell_plot_df[repr_col] = (cell_plot_df[conc_col]
                                  / cell_plot_df['total_conc'])
        cell_plot_df[H_col] = cell_plot_df.apply(
            lambda df: entropy(df[conc_col]), axis=1)
        cell_plot_df[KL_col] = cell_plot_df.apply(
            lambda df: kl_div(df[conc_col], df['total_conc']), axis=1)

        # We save the column names in grps_dict
        grp_dict['conc_col'] = conc_col
        grp_dict['prop_col'] = prop_col
        grp_dict['repr_col'] = repr_col
        grp_dict['H_col'] = H_col
        grp_dict['KL_col'] = KL_col
        grp_label = grp_dict['grp_label']
        grp_dict['prop_label'] = f'Proportion of {grp_label} in the cell'
        grp_dict['conc_label'] = f'Concentration of {grp_label} in the cell'
        grp_dict['repr_label'] = f'Representation of {grp_label} in the cell'
        grp_dict['H_label'] = f'Concentration entropy of {grp_label} in the cell'
        grp_dict['KL_label'] = f'KL divergence of {grp_label} in the cell'
        grps_dict[grp] = grp_dict
    return cell_plot_df, grps_dict


def all_grps_metric(metric_dict, cell_plot_df, grps_dict):
    '''
    Computes the metric described in `metric_dict` from the data in
    `cell_plot_df` for each group described in `grps_dict`, and the weighted
    average over all groups.
    '''
    metric_readable = metric_dict['readable']
    N = cell_plot_df['total_count'].sum()
    max_metric_all = 0
    metric_all = 0
    for grp, grp_dict in grps_dict.items():
        grp_label = grp_dict['grp_label']
        count_col = grp_dict['count_col']
        metric_col = grp_dict[metric_dict['name'] + '_col']
        N_grp = cell_plot_df[count_col].sum()
        metric_grp = cell_plot_df[metric_col].abs().sum()
        prop_grp = N_grp / N
        max_metric_grp = metric_dict['max_fun'](cell_plot_df, count_col)
        normed_metric_grp = metric_grp / max_metric_grp
        print(f'The {metric_readable} for the {grp_label} is: '
              f'{normed_metric_grp}.')
        max_metric_all += prop_grp * max_metric_grp
        metric_all += prop_grp * metric_grp

    print(f'The {metric_readable} averaged over all groups is '
          f'{metric_all / max_metric_all}.')
    return metric_all, max_metric_all


def clusters(vectors, max_nr_clusters=10, plot=True, random_state=0):
    '''
    Performs clusters analyses based on the vectors contained in `vectors` for
    a number of clusters varying from 1 to `max_nr_clusters`.
    '''
    all_vars = []
    all_cells_clusters = []
    all_clusters_centers = []

    for n_clusters in range(1, max_nr_clusters+1):
        model = KMeans(
            n_clusters=n_clusters, random_state=random_state).fit(vectors)
        var = model.inertia_
        all_vars.append(var)
        cells_clusters = model.labels_
        all_cells_clusters.append(cells_clusters)
        clusters_centers = model.cluster_centers_
        all_clusters_centers.append(clusters_centers)

    all_vars = np.array(all_vars)
    ax = None
    if plot:
        ax = helpers_viz.cluster_analysis(all_vars)

    return all_vars, all_cells_clusters, all_clusters_centers, ax


def ks_test_2d(cell_plot_df, obs_col, pred_col, Nx, Ny, n_samples):
    '''
    Performs a 2D KS-test to quantify the discrepancy between the probability
    distributions found in the columns `obs_col` and `pred_col` of
    `cell_plot_df`. It is assumed here that the cells are ordered by column
    and then by row (in other words, we go down columns as the index increases).
    '''
    indices_cells = cell_plot_df['cell_id'].values
    proba_in_cells = cell_plot_df[obs_col].values
    # We create a matrix representing all the cells, including those out of
    # the area of interest, for simplicity. The latter will keep a null value,
    # so they won't have any influence on the value of the CDF.
    proba_in_all_cells = np.zeros(Nx*Ny)
    proba_in_all_cells[indices_cells] = proba_in_cells
    top_left_proba_matrix = proba_in_all_cells.reshape((Nx, Ny)).T
    # We then create a list with the x-, y-, and x,y-flipped probability
    # matrices, so as to calculate the CDF starting from all four corners.
    y_flipped_proba_matrix = np.flip(top_left_proba_matrix, axis=0)
    list_proba_matrices = [
        top_left_proba_matrix, np.flip(top_left_proba_matrix, axis=1),
        y_flipped_proba_matrix, np.flip(y_flipped_proba_matrix, axis=1)]
    # Since we flipped the matrices, the cumulative sums are always performed
    # in the same way, from top-left to bottom-right.
    list_obs_cdf = []
    for proba_matrix in list_proba_matrices:
        col_cumsum = proba_matrix.cumsum(axis=0)
        # We calculate the cdf matrix Pij = p(x < x_i, y < y_j)
        list_obs_cdf.append(col_cumsum.cumsum(axis=1))

    # We then do the same thing for the model distribution.
    pred_proba_in_cells = cell_plot_df[pred_col].values
    pred_proba_in_all_cells = np.zeros(Nx*Ny)
    pred_proba_in_all_cells[indices_cells] = pred_proba_in_cells
    pred_proba_matrix = pred_proba_in_all_cells.reshape((Nx, Ny)).T
    y_flipped_proba_matrix = np.flip(pred_proba_matrix, axis=0)
    list_pred_proba_matrices = [
        pred_proba_matrix, np.flip(pred_proba_matrix, axis=1),
        y_flipped_proba_matrix, np.flip(y_flipped_proba_matrix, axis=1)]

    list_pred_cdf = []
    for proba_matrix in list_pred_proba_matrices:
        col_cumsum = proba_matrix.cumsum(axis=0)
        list_pred_cdf.append(col_cumsum.cumsum(axis=1))

    # We're now able to calculate the maximum differences between the model and
    # measured CDFs, starting from all four corners.
    list_delta_cdf = []
    for i in range(4):
        delta_cdf = np.max(np.abs(list_pred_cdf[i] - list_obs_cdf[i]))
        list_delta_cdf.append(delta_cdf)
    print(list_delta_cdf)
    # Our KS score is then simply the maximum of the four obtained distances.
    ks_score = max(list_delta_cdf)
    # We finally calculate the p-value, to check if the distance obtained
    # between the distribution is significant enough not to be due to chance
    # (see Peacock, 1983 for reference).
    Z = np.sqrt(n_samples) * ks_score
    Z_inf = Z / (1 - 0.53*n_samples**(-0.9))
    p_value = 2 * np.exp(-2 * (Z_inf-0.5)**2)
    return ks_score, p_value
