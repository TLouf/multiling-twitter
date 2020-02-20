import matplotlib.pyplot as plt
import src.utils.scales as scales
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
    vmin, vmax = scales.get_global_vmin_vmax(cell_plot_df, metric_dict,
                                             grps_dict, min_count=min_count)


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
