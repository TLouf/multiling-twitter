import matplotlib.pyplot as plt
import numpy as np
import src.visualization.grid_viz as grid_viz

def cluster_analysis(all_vars, max_nr_clusters=10, show=True):
    '''
    Plots the evolution of the variance explained when adding clusters, from an
    array `all_vars` containing the variances obtained from applying a cluster
    algorithm, sorted in ascending number of clusters, from 1 to
    `max_nr_clusters`.
    '''
    max_var = all_vars.max()
    x_plot = range(1, max_nr_clusters+1)
    y_plot = 1 - all_vars / max_var
    plt.plot(x_plot, y_plot, marker='.')
    plt.xlabel('number of clusters')
    plt.ylabel('proportion of variance explained')
    ax = plt.gca()
    if show:
        plt.show()
    plt.close()
    return ax


def metric_grid(cell_plot_df, metric_dict, shape_df, grps_dict, country_name,
                cmap='coolwarm', save_path_format=None, xy_proj='epsg:3857',
                min_count=0, null_color='None'):
    '''
    Plots the metric described in `metric_dict` for every group described in
    `grps_dict` from the cell data contained in `cell_plot_df`. In fact, it
    simply wraps `grid_viz.plot_grid` and feeds it the right data for every
    group, getting appropriate labels, colorbar scale and save path (if any).
    '''
    # We find the minimum and maximum values of the representation found for
    # every group. This way we keep the same scale across the graphs, enabling
    # better comparison.
    metric = metric_dict['name']
    log_scale = metric_dict['log_scale']
    readable_metric = metric_dict['readable']
    count_mask = cell_plot_df[metric_dict['total_count_col']] > min_count
    vmin, vmax = get_global_vmin_vmax(cell_plot_df, metric_dict, grps_dict,
                                      min_count=min_count)

    for grp, grp_dict in grps_dict.items():
        readable_lang = grp_dict['readable']
        grp_label = grp_dict['grp_label']
        plot_title = f'{readable_metric} of {grp_label} in {country_name}'
        metric_col = grp_dict[metric + '_col']
        cbar_label = f'{readable_metric} of {grp_label} in the cell'
        if save_path_format:
            save_path = save_path_format.format(grp=grp)
        else:
            save_path = None
        # The cells with a count not relevant enough will simply not be plotted,
        # they'll have the background color.
        plot_kwargs = dict(edgecolor='w', linewidths=0.2, cmap=cmap)
        ax_prop = grid_viz.plot_grid(
            cell_plot_df.loc[count_mask], shape_df, metric_col=metric_col,
            save_path=save_path, title=plot_title, cbar_label=cbar_label,
            xy_proj=xy_proj, log_scale=log_scale, vmin=vmin, vmax=vmax,
            null_color=null_color, **plot_kwargs)


def get_global_vmin_vmax(cell_plot_df, metric_dict, grps_dict, min_count=0):
    '''
    We find the minimum and maximum values of the metric found for every group,
    and if the metric requires it, we make the obtained scale symmetric about
    (or centered around) a certain value. This way we keep the same scale across
    the graphs, enabling better comparison.
    '''
    log_scale = metric_dict['log_scale']
    vmin = metric_dict.get('vmin')
    vmax = metric_dict.get('vmax')
    if (vmin is None) or (vmax is None):
        # We initialize with plus and minus infinity
        vmin = np.inf
        vmax = -np.inf
        for grp, grp_dict in grps_dict.items():
            grp_metric_col = grp_dict[metric_dict['name'] + '_col']
            vmin_col, vmax_col = get_col_vmin_vmax(cell_plot_df, metric_dict,
                                                   col=grp_metric_col,
                                                   min_count=min_count)
            vmin = min(vmin, vmin_col)
            vmax = max(vmax, vmax_col)

        sym_about = metric_dict.get('sym_about')
        if sym_about is not None:
            vmin, vmax = get_sym_about(vmin, vmax, sym_about,
                                       log_scale=log_scale)

    return vmin, vmax


def get_col_vmin_vmax(cell_plot_df, metric_dict, col=None, min_count=0):
    '''
    Get the (relevant) minimum and maxmimum value in the column of `cell_pot_df`
    corresponding to the metric described in `metric_dict`, or in the manually
    specified `col`. The relevance is relative to a minimum value `min_count` of
    the column `metric_dict['total_count_col']`.
    '''
    total_count_col = metric_dict['total_count_col']
    log_scale = metric_dict['log_scale']
    if col is None:
        col = metric_dict['name']

    count_mask = cell_plot_df[total_count_col] > min_count
    if log_scale:
        vmin_mask = count_mask & (cell_plot_df[col] > 0)
    else:
        vmin_mask = count_mask
    # We take the minimum among the relevant cells (count_mask) and which is
    # not 0 if we are in log scale.
    vmin_col = cell_plot_df.loc[vmin_mask, col].min()
    vmax_col = cell_plot_df.loc[count_mask, col].max()
    return vmin_col, vmax_col


def get_sym_about(vmin, vmax, sym_about, log_scale=False):
    '''
    Finds the most extreme value among `vmin` and `vmax`, with respect to
    `sym_about` (on a linear or logarithmic scale) and takes the other extrema
    as the inverse of this most extreme value, so as to have a scale symmetric
    about `sym_about`.
    '''
    if log_scale:
        if sym_about/vmin > vmax/sym_about:
            vmax = 1/vmin * sym_about**2
        else:
            vmin = 1/vmax * sym_about**2
    else:
        if sym_about-vmin > vmax-sym_about:
            vmax = 2*sym_about - vmin
        else:
            vmin = 2*sym_about - vmax
    return vmin, vmax
