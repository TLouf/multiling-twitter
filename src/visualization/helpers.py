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


def repr_grid(cell_plot_df, shape_df, grps_dict, country_name, cmap='coolwarm',
              save_path=None, xy_proj='epsg:3857', min_count=0):
    '''
    Plots the representation of every group described in `grps_dict` from the
    cell data contained in `cell_plot_df`. The colormap `cmap` should be a
    diverging one, but not white at the middle value, as white is reserved
    for cells with a null total count.
    '''
    # We find the minimum and maximum values of the representation found for
    # every group. This way we keep the same scale across the graphs, enabling
    # better comparison.
    vmin = 1
    vmax = 1
    for grp, plot_dict in grps_dict.items():
        repr_col = plot_dict['repr_col']
        count_mask = cell_plot_df['total_count'] > min_count
        repr_mask = cell_plot_df[repr_col] > 0
        # We take the minimum among the relevant cells (count_mask) and which is
        # not 0 (repr_mask).
        vmin_lang = cell_plot_df.loc[count_mask & repr_mask, repr_col].min()
        vmin = min(vmin, vmin_lang)
        vmax_lang = cell_plot_df.loc[count_mask, repr_col].max()
        vmax = max(vmax, vmax_lang)

    # We take the most extreme value for the range of the colorbar, and take
    # the other extrema as the inverse of this most extreme value, so as to
    # have a colorbar symmetric about 1.
    if vmin < 1/vmax:
        vmax = 1/vmin
    else:
        vmin = 1/vmax

    for grp, plot_dict in grps_dict.items():
        readable_lang = plot_dict['readable']
        grp_label = plot_dict['grp_label']
        plot_title = f'Representation of {grp_label} in {country_name}'
        repr_col = plot_dict['repr_col']
        cbar_label = plot_dict['repr_label']
        # The cells with a count not relevant enough will simply not be plotted,
        # they'll have the background color.
        count_mask = cell_plot_df['total_count'] > min_count
        plot_kwargs = dict(edgecolor='w', linewidths=0.2, cmap=cmap)
        ax_prop = grid_viz.plot_grid(
            cell_plot_df.loc[count_mask], shape_df, metric_col=repr_col,
            save_path=save_path, title=plot_title, cbar_label=cbar_label,
            xy_proj=xy_proj, log_scale=True, vmin=vmin, vmax=vmax,
            **plot_kwargs)
