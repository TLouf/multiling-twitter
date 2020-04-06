import os
import matplotlib.pyplot as plt
import src.utils.scales as scales
import src.visualization.grid_viz as grid_viz


def top_lang_speakers(user_langs_agg, area_dict, lang_relevant_count,
                      lang_relevant_prop, fig_dir=None, show=False):
    '''
    Produces bar plots of the top ten languages in the area in terms of number
    of users and proportion. The data comes from a user-aggregate level, in
    `user_langs_agg`, which lists all users and the languages each speaks.
    '''
    area_name = area_dict['readable']
    cc = area_dict['cc']
    # Get the number of users speaking every language, and sort the languages
    # starting with the most spoken.
    area_langs_counts = (user_langs_agg.groupby('cld_lang')
                                       .size()
                                       .rename('count')
                                       .sort_values(ascending=False))
    total_count = len(user_langs_agg.index.levels[0])
    # Then take the top ten languages.
    top_langs = area_langs_counts.index.values[:10]
    top_counts = area_langs_counts.values[:10]

    plt.bar(top_langs, top_counts)
    plt.title(f'Ten languages with the most speakers in {area_name}')
    plt.ylabel('number of speakers')
    if fig_dir:
        file_name = (
            f'top_langs_speakers_count_{area_name}_count_th='
            f'{lang_relevant_count}_prop_th={lang_relevant_prop}.pdf')
        save_dir = os.path.join(fig_dir, cc)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()

    plt.bar(top_langs, top_counts/total_count)
    plt.title(f'Ten languages with the most speakers in {area_name}')
    plt.ylabel('proportion of the users speaking')
    if fig_dir:
        file_name = (
            f'top_langs_speakers_prop_{area_name}_count_th='
            f'{lang_relevant_count}_prop_th={lang_relevant_prop}.pdf')
        save_dir = os.path.join(fig_dir, cc)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()


def ling_grps(multiling_grps, ling_counts, total_count, area_dict,
              lang_relevant_count, lang_relevant_prop,
              fig_dir=None, show=False):
    '''
    Produces bar plots of the top ten linguals groups in the area in terms of
    number of users and proportion. The data comes directly from `ling_counts`,
    which has the counts for every group in `multiling_grps`.
    '''
    area_name = area_dict['readable']
    cc = area_dict['cc']
    x_plot = [grp[5:] for grp in multiling_grps]
    plt.bar(x_plot, ling_counts)
    plt.title(f'Local languages groups in {area_name}')
    plt.ylabel('number in the group')
    if fig_dir:
        file_name = (
            f'multilinguals_count_{area_name}_count_th={lang_relevant_count}'
            f'_prop_th={lang_relevant_prop}.pdf')
        save_dir = os.path.join(fig_dir, cc)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()

    plt.bar(x_plot, ling_counts/total_count)
    plt.title(f'Local languages groups in {area_name}')
    plt.ylabel('proportion out of the total population')
    if fig_dir:
        file_name = (
            f'multilinguals_prop_{area_name}_count_th={lang_relevant_count}'
            f'_prop_th={lang_relevant_prop}.pdf')
        save_dir = os.path.join(fig_dir, cc)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()


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
        grp_label = grp_dict['grp_label']
        plot_title = f'{readable_metric} of {grp_label} in {country_name}'
        cbar_label = f'{readable_metric} of {grp_label} in the cell'
        # If the metric is already over all groups on the cell level,
        # then we'll have a single group in grp_dict and the column in
        # `cell_plot_df` will simply be the name of the metric.
        grp_metric_col = grp_dict.get(metric_dict['name'] + '_col') or metric
        if save_path_format:
            save_path = save_path_format.format(grp=grp)
        else:
            save_path = None
        # The cells with a count not relevant enough will simply not be plotted,
        # they'll have the background color.
        plot_kwargs = dict(edgecolor='w', linewidths=0.2, cmap=cmap)
        grid_viz.plot_grid(
            cell_plot_df.loc[count_mask], shape_df, metric_col=grp_metric_col,
            save_path=save_path, title=plot_title, cbar_label=cbar_label,
            xy_proj=xy_proj, log_scale=log_scale, vmin=vmin, vmax=vmax,
            null_color=null_color, **plot_kwargs)
