from collections import defaultdict
from scipy.stats import chisquare
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ortools.linear_solver import pywraplp
import src.visualization.helpers as helpers_viz
import src.utils.geometry as geo

def kl_div(obs, pred):
    '''
    Computes -obs*log(pred/obs), the elements over which one sums to get the
    Kullback Leibler divergence between the distribution defined by all the obs
    and the one defined by all the pred. pred == 0 must imply obs == 0.
    '''
    # We add these two steps to avoid a 0*log(0), as in Python 0**0 = 1, but
    # 0*log(0) is not a number.
    den = obs**obs
    num = pred**obs
    return -np.log(num / den)


def max_kl(cell_plot_df, grp_count_col):
    '''
    Computes the maximum KL divergence value that a given group can reach in
    a cell.
    '''
    N = cell_plot_df['total_count'].sum()
    N_grp = cell_plot_df[grp_count_col].sum()
    return -np.log(N_grp / N)


def entropy(p, cell_size=10000):
    '''
    Computes -p*log(p), the elements over which one sums to get Shannon's
    entropy of a system defined by the probability distribution {p}.
    '''
    rel_cell_area = (cell_size / 10000)**2
    # Written this way to have 0*log 0 = 1.
    return -np.log((p / rel_cell_area)**p)


def null_Hc(cell_plot_df, grp_count_col, n_iters=100, cell_size=None):
    '''
    Returns the null model concentration entropy for a group described in
    `grp_dict`. A random null model is iterated `n_iters` times to reproduce the
    finite size effect, which is relevant here as the number of groups of the
    multinomial is the number of cells, which can be hundreds or even thousands.
    Thus with this method the value of the null model Hc is different for each
    group, as they have different sizes and thus require each a different
    random null model.
    '''
    if cell_size is None:
        cell_size = (cell_plot_df.area)**0.5
    N_grp = cell_plot_df[grp_count_col].sum()
    distrib = cell_plot_df['total_conc'].values
    cells_pop = np.random.multinomial(int(N_grp), distrib, n_iters)
    conc_pred = cells_pop / N_grp
    if isinstance(cell_size, pd.Series):
        cell_size = np.broadcast_to(cell_size.values, conc_pred.shape)
    return entropy(conc_pred, cell_size=cell_size).sum(axis=1).mean()


def null_Hp(cell_plot_df, grps_dict):
    '''
    Returns the null model proportion entropy for the system consisting of the
    groups described in `grps_dict`, with the cell counts in `cell_plot_df`. A
    random null model doesn't make sense here, because the finite size effect
    is irrelevant when you have ~10 groups at most.
    '''
    Hp_null = 0
    sum_local_counts = cell_plot_df['local_count'].sum()
    for _, grp_dict in grps_dict.items():
        count_col = grp_dict['count_col']
        p_grp = cell_plot_df[count_col].sum() / sum_local_counts
        Hp_null += entropy(p_grp)
    return Hp_null


def calc_by_cell(cell_plot_df, grps_dict, cell_size=None):
    '''
    Adds columns to cell_plot_df with multiple metrics of interest, for every
    group described in grps_dict.
    '''
    total_nr_users = cell_plot_df['total_count'].sum()
    cell_plot_df['total_conc'] = cell_plot_df['total_count'] / total_nr_users
    Hp_col = 'Hp'
    cell_plot_df[Hp_col] = 0
    if cell_size is None:
        cell_size = (cell_plot_df.area)**0.5

    for grp, grp_dict in grps_dict.items():
        count_col = grp_dict['count_col']
        conc_col = f'conc_{grp}'
        prop_col = f'prop_{grp}'
        repr_col = f'repr_{grp}'
        Hc_col = f'Hc_{grp}'
        KL_col = f'KL_{grp}'

        grp_total = cell_plot_df[count_col].sum()
        cell_plot_df[conc_col] = cell_plot_df[count_col] / grp_total
        cell_plot_df[prop_col] = (cell_plot_df[count_col]
                                  / cell_plot_df['local_count'])
        cell_plot_df[repr_col] = (cell_plot_df[conc_col]
                                  / cell_plot_df['total_conc'])
        cell_plot_df[Hc_col] = entropy(cell_plot_df[conc_col],
                                       cell_size=cell_size)
        cell_plot_df[Hp_col] += entropy(cell_plot_df[prop_col])
        cell_plot_df[KL_col] = kl_div(cell_plot_df[conc_col],
                                      cell_plot_df['total_conc'])

        # We save the column names in grps_dict
        grp_dict['conc_col'] = conc_col
        grp_dict['prop_col'] = prop_col
        grp_dict['repr_col'] = repr_col
        grp_dict['Hc_col'] = Hc_col
        grp_dict['KL_col'] = KL_col
        grp_label = grp_dict['grp_label']
        grp_dict['prop_label'] = f'Proportion of {grp_label} in the cell'
        grp_dict['conc_label'] = f'Concentration of {grp_label} in the cell'
        grp_dict['repr_label'] = f'Representation of {grp_label} in the cell'
        grp_dict['Hc_label'] = (
            f'Concentration entropy of {grp_label} in the cell')
        grp_dict['KL_label'] = f'KL divergence of {grp_label} in the cell'
        grps_dict[grp] = grp_dict

    Hp_null = null_Hp(cell_plot_df, grps_dict)
    cell_plot_df[Hp_col] = cell_plot_df[Hp_col] / Hp_null
    return cell_plot_df, grps_dict


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
        cdf_matrix = col_cumsum.cumsum(axis=1)
        list_obs_cdf.append(cdf_matrix)

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
        pred_cdf_matrix = col_cumsum.cumsum(axis=1)
        list_pred_cdf.append(pred_cdf_matrix)

    # We're now able to calculate the maximum differences between the model and
    # measured CDFs, starting from all four corners.
    list_delta_cdf = []
    for i in range(4):
        delta_cdf = np.max(np.abs(list_pred_cdf[i] - list_obs_cdf[i]))
        list_delta_cdf.append(delta_cdf)
    # Our KS score is then simply the maximum of the four distances we computed.
    ks_score = max(list_delta_cdf)
    # We finally calculate the p-value, to check if the distance obtained
    # between the distribution is significant enough not to be due to chance
    # (see Peacock, 1983 for reference).
    Z = np.sqrt(n_samples) * ks_score
    Z_inf = Z / (1 - 0.53*n_samples**(-0.9))
    p_value = 2 * np.exp(-2 * (Z_inf-0.5)**2)
    return ks_score, p_value


def grid_chisquare(cell_plot_df, obs_col, pred_col, n_samples):
    '''
    Computes the chi square score and the associated p value, to check if the
    cell distribution in the `obs_col` column of `cell_plot_df` may have been
    drawn from the distribution defined in the `pred_col` column.
    '''
    # number of cells with non null total count obviously
    f_exp = cell_plot_df[pred_col].values
    f_obs = cell_plot_df[obs_col].values
    # degree of freedom equals to (numbers of cells-1) * (1 -1), so ddof
    # in scipy's chisquare is the default, 0.
    chi2_score, p_value = chisquare(n_samples*f_obs, n_samples*f_exp)
    return chi2_score, chi2_score/n_samples, p_value


def earthmover_distance(cell_plot_df, dist1_dict, dist2_dict, d_matrix=None):
    '''
    Computes the EMD between the concentration distributions described by the
    dictionaries `dist1_dict` and `dist2_dict`, and whose data are comprised
    within the columns of `cell_plot_df`. Also returns a norm, defined as the
    average distance between cells, weighted by the minimum population in each
    pair.
    '''
    if d_matrix is None:
        d_matrix = geo.d_matrix_from_cells(cell_plot_df)
    solver = pywraplp.Solver('earthmover_distance',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    variables = dict()
    # for each pile in dist1, the constraint that says all the dirt must leave
    # this pile
    dirt_leaving_constraints = defaultdict(lambda: 0)
    # for each hole in dist2, the constraint that says this hole must be filled
    dirt_filling_constraints = defaultdict(lambda: 0)
    # The objective is initialized as a minimization.
    objective = solver.Objective()
    objective.SetMinimization()
    n_cells = cell_plot_df.shape[0]
    for i in range(n_cells):
        for j in range(n_cells):
            # We declare the variables of the optimization problem, the flows
            # from cells i to j, and constrain them to be positive.
            flow_i_j = solver.NumVar(
                0, solver.infinity(), 'z_{%s, %s}' % (i, j))
            variables[(i, j)] = flow_i_j
            dirt_leaving_constraints[i] += flow_i_j
            dirt_filling_constraints[j] += flow_i_j
            # The function to minimize is the sum of flow_i_j times
            # d_matrix[i,j], so to each variable flow_i_j we set
            # the corresponding distance as a coefficient.
            objective.SetCoefficient(flow_i_j, d_matrix[i, j])

    # We add the constraints that sum over j of flow_i_j is equal to
    # dist1[i]. This is a strict equality here because dist1 and dist2 are
    # probability distributions (see Levina and Bickel, 2001).
    dist1 = cell_plot_df[dist1_dict['conc_col']].values
    dist1 = dist1 / dist1.sum()
    for i, linear_combination in dirt_leaving_constraints.items():
        solver.Add(linear_combination == dist1[i])

    dist2 = cell_plot_df[dist2_dict['conc_col']].values
    dist2 = dist2 / dist2.sum()
    for j, linear_combination in dirt_filling_constraints.items():
        solver.Add(linear_combination == dist2[j])

    status = solver.Solve()
    if status not in [solver.OPTIMAL, solver.FEASIBLE]:
        raise Exception('Unable to find feasible solution')
    emd_value = objective.Value()

    norm = 0
    total_weight = 0
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            weight = min(cell_plot_df['total_conc'].values[i],
                         cell_plot_df['total_conc'].values[j])
            total_weight += weight
            norm += weight * d_matrix[i, j]
    norm = norm / total_weight
    grp1_label = dist1_dict['grp_label']
    grp2_label = dist2_dict['grp_label']
    print(f'The EMD between the distributions of {grp1_label} and of '
          f'{grp2_label} is {emd_value/norm}.')
    return emd_value, norm, d_matrix


def all_grps_metric(metric_dict, cell_plot_df, grps_dict, **scale_fun_kwargs):
    '''
    Computes the metric described in `metric_dict` from the data in
    `cell_plot_df` for each group described in `grps_dict`, and the weighted
    average over all groups.
    '''
    metric_readable = metric_dict['readable']
    scale_fun = metric_dict['scale_fun']
    sym_about = metric_dict.get('sym_about') or 0
    N = 0
    for grp, grp_dict in grps_dict.items():
        count_col = grp_dict['count_col']
        N += cell_plot_df[count_col].sum()
    scale_metric_all = 0
    metric_all = 0
    scaled_metric_all = 0
    for grp, grp_dict in grps_dict.items():
        grp_label = grp_dict['grp_label']
        count_col = grp_dict['count_col']
        metric_col = grp_dict[metric_dict['name'] + '_col']
        N_grp = cell_plot_df[count_col].sum()
        prop_grp = N_grp / N
        metric_grp = cell_plot_df[metric_col].abs().sum()
        if metric_dict.get('global_norm'):
            scaled_metric_grp = metric_grp
            scale_metric_all += prop_grp * scale_fun(
                cell_plot_df, count_col, **scale_fun_kwargs)
            metric_all += prop_grp * metric_grp
        else:
            scale_metric_grp = scale_fun(cell_plot_df, count_col,
                                         **scale_fun_kwargs)
            scaled_metric_grp = abs(sym_about - metric_grp / scale_metric_grp)
            scaled_metric_all += prop_grp * scaled_metric_grp
            print(metric_grp, scale_metric_grp)
        grp_dict[metric_dict['name']] = scaled_metric_grp
        grps_dict[grp] = grp_dict
        print(f'The {metric_readable} for the {grp_label} is: '
              f'{scaled_metric_grp}.')

    if metric_dict.get('global_norm'):
        scaled_metric_all = abs(sym_about - metric_all / scale_metric_all)
    print(f'The {metric_readable} averaged over all groups is '
          f'{scaled_metric_all}.')
    return scaled_metric_all, grps_dict


def all_cells_metric(metric_dict, cell_plot_df):
    '''
    Computes the metric described in `metric_dict` from the data in
    `cell_plot_df` averaged over all cells.
    '''
    metric_readable = metric_dict['readable']
    metric_col = metric_dict['name']
    sym_about = metric_dict.get('sym_about') or 0
    total_count_col = metric_dict['total_count_col']
    N = cell_plot_df[total_count_col].sum()
    N_cell = cell_plot_df[total_count_col]
    cell_metric = abs(sym_about - cell_plot_df[metric_col]) * N_cell / N
    global_metric = cell_metric.sum()
    print(f'The {metric_readable} averaged over all cells is {global_metric}')
    return global_metric
