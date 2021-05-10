import os
import numpy as np
import matplotlib.pyplot as plt
import src.utils.scales as scales
import src.visualization.grid_viz as grid_viz
from adjustText import adjust_text

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
    plt.tight_layout()
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
    plt.tight_layout()
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
    plt.tight_layout()
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
    plt.tight_layout()
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


def metric_grid(cell_plot_df, metric_dict, shape_df, grps_dict,
                save_path_format=None, min_count=0, **plot_grid_kwargs):
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
    plot_grid_kwargs.setdefault('log_scale', metric_dict['log_scale'])
    readable_metric = metric_dict['readable']
    count_mask = cell_plot_df[metric_dict['total_count_col']] > min_count
    vmin, vmax = scales.get_global_vmin_vmax(cell_plot_df, metric_dict,
                                             grps_dict, min_count=min_count)
    vmin = metric_dict.get('vmin', vmin)
    vmax = metric_dict.get('vmax', vmax)
    plot_grid_kwargs.setdefault('vmin', vmin)
    plot_grid_kwargs.setdefault('vmax', vmax)
    cmap = metric_dict.get('cmap')
    if cmap is None:
        if metric_dict.get('sym_about') is None:
            cmap = 'plasma'
        else:
            cmap = 'bwr'
    plot_kwargs = {'plot': {
        'edgecolor': (0.9, 0.9, 0.9), 'linewidths': 0.1, 'cmap': cmap}}

    for grp, grp_dict in grps_dict.items():
        grp_label = grp_dict['grp_label']
        cbar_label = f'{readable_metric} of {grp_label}'
        # Have to make a copy or else the first group's value will be conserved.
        grp_plot_grid_kwargs = plot_grid_kwargs.copy()
        grp_plot_grid_kwargs.setdefault('cbar_label', cbar_label)
        # If the metric is already over all groups on the cell level,
        # then we'll have a single group in grp_dict and the column in
        # `cell_plot_df` will simply be the name of the metric.
        grp_metric_col = grp_dict.get(metric_dict['name'] + '_col', metric)
        if save_path_format:
            save_path = save_path_format.format(grp=grp)
            save_dir = os.path.split(save_path)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_path = None
        # The cells with a count not relevant enough will simply not be plotted,
        # they'll have the background color.
        grid_viz.plot_grid(
            cell_plot_df.loc[count_mask], shape_df, metric_col=grp_metric_col,
            save_path=save_path, **plot_kwargs, **grp_plot_grid_kwargs)


def axis_config(ax_dict, config, min_count=0):
    '''
    Convenience function to fill in a dictionary `ax_dict` of the kwargs to pass
    to `grid_viz.plot_grid`, calculating th suiting vmin and vmax of the
    colorbar, generating its label...
    '''
    cell_df = config['cell_plot_df']
    grps_dict = config['grps_dict']
    metric_dict = config['metric_dict']
    metric = metric_dict['name']
    ax_dict['log_scale'] = metric_dict['log_scale']
    readable_metric = metric_dict['readable']
    vmin, vmax = scales.get_global_vmin_vmax(cell_df, metric_dict,
                                             grps_dict, min_count=min_count)
    vmin = metric_dict.get('vmin', vmin)
    vmax = metric_dict.get('vmax', vmax)
    ax_dict['vmin'] = vmin
    ax_dict['vmax'] = vmax

    cmap = metric_dict.get('cmap')
    if cmap is None:
        if metric_dict.get('sym_about') is None:
            cmap = 'plasma'
        else:
            cmap = 'bwr'
    ax_dict['plot']['cmap'] = cmap
    grp_dict = grps_dict[config['grp']]
    ax_dict['metric_col'] = grp_dict.get(metric_dict['name'] + '_col', metric)
    cbar_label = metric_dict.get('cbar_label')
    if cbar_label is None:
        grp_label = grp_dict['grp_label']
        cbar_label = f'{readable_metric} of {grp_label}'
    ax_dict['cbar_label'] = cbar_label
    return ax_dict


def scatter_labelled(x_plot, y_plot, labels, xlabel=None, ylabel=None,
                     cmap='jet', figsize=None, annot=True, lgd=None,
                     save_path=None, show=True, fig=None, ax=None,
                     tight_layout=True, pts_s=6, **kwargs):
    '''
    Scatter plot using the data (`x_plot`, `y_plot`) and labels `labels` for
    each point. These labels are either assigned to a color in a legend, or
    directly added as annotations to the data points. The annotations are
    placed with `adjust_text` to avoid overlapping.
    '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    n_pts = len(labels)
    # Sort x_plot, y_plot and labels by ascending x_plot values.
    sorted_lists = sorted(zip(x_plot, y_plot, labels), key=lambda t: t[0])
    x_plot, y_plot, labels = zip(*sorted_lists)
    colors = plt.cm.get_cmap(cmap, n_pts)

    for i, lab in enumerate(labels):
        ax.scatter(x_plot[i], y_plot[i], label=lab, c=(colors(i),), s=pts_s)

    if annot:
        # Add annotations to each data point, placed on the point directly.
        texts = [ax.text(x_plot[i], y_plot[i], labels[i]) for i in range(n_pts)]
        adj_text_kw = {
            'only_move': {'points': 'xy', 'text': 'y', 'objects': 'xy'},
            'arrowprops': {'arrowstyle': '-', 'color': 'gray', 'lw': 0.1}}
        adj_text_kw.update(kwargs.get('adj_text', {}))
        # Iteratively find a better placement for annotations, avoiding overlap.
        adjust_text(texts, x=x_plot, y=y_plot, ax=ax, **adj_text_kw)
    bbox_extra_artists = ()
    if lgd:
        # Place a legend outside the graph, in the top left corner.
        lgd_kwargs = {'bbox_to_anchor': (1.05, 1), 'loc': 2,
                      'borderaxespad': 0, 'ncol': 1}
        lgd_kwargs.update(kwargs.get('lgd', {}))
        lgd = ax.legend(**lgd_kwargs)
        bbox_extra_artists = (lgd,)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.set_tight_layout(tight_layout)
    if save_path:
        fig.savefig(save_path, bbox_extra_artists=bbox_extra_artists,
                    bbox_inches='tight')
    if show:
        fig.show()
    return fig, ax


def errorbar_labelled(x_plot, y_plot, labels, xlabel=None, ylabel=None,
                      show_xerr=True, show_yerr=True,
                      cmap='jet', figsize=None, annot=True, lgd=None,
                      save_path=None, show=True, fig=None, ax=None,
                      tight_layout=True, pts_s=6, **kwargs):
    '''
    Errorbar plot using the data (`x_plot`, `y_plot`) and labels `labels` for
    each point. `x_plot` and `y_plot` should each be a list containing for
    each point the array of values corresponding to each label. The function
    plots the point corresponidng to the mean of these arrays with bars going
    from the lowest to the largest value contained in the array, thus showing
    the range of values reached, each value being shown as a smaller point, with
    a cross. The labels are either assigned to a color in a legend, or directly
    added as annotations to the data points. The annotations are placed with
    `adjust_text` to avoid overlapping.
    '''
    scatter_kwargs = kwargs.get('scatter', {'s': 6**2, 'marker': '+'})
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    n_pts = len(labels)
    x_mean = [x.mean() for x in x_plot]
    y_mean = [y.mean() for y in y_plot]
    # Sort data by ascending x_mean values for coloring.
    sorted_lists = sorted(zip(x_mean, y_mean, x_plot, y_plot, labels),
                          key=lambda t: t[0])
    x_mean, y_mean, x_plot, y_plot, labels = zip(*sorted_lists)
    xerr = [None] * n_pts
    if show_xerr:
        xerr = [np.array([[np.max(x_mean[i] - x_plot[i])],
                          [np.max(x_plot[i] - x_mean[i])]])
                for i in range(n_pts)]
    yerr = [None] * n_pts
    if show_yerr:
        yerr = [np.array([[np.max(y_mean[i] - y_plot[i])],
                          [np.max(y_plot[i] - y_mean[i])]])
                for i in range(n_pts)]
    colors = plt.cm.get_cmap(cmap, n_pts)
    for i in range(n_pts):
        c = colors(i)
        _, _, barlinecols = ax.errorbar(
            x_mean[i], y_mean[i], xerr=xerr[i], yerr=yerr[i],
            ls='', marker='o', ms=pts_s, c=c, label=labels[i])
        for b in barlinecols:
            b.set_linestyle(':')
        if show_xerr:
            y_of_x_pts = y_mean[i].repeat(len(x_plot[i]))
            ax.scatter(x_plot[i], y_of_x_pts, c=(c,), **scatter_kwargs)
        if show_yerr:
            x_of_y_pts = x_mean[i].repeat(len(y_plot[i]))
            ax.scatter(x_of_y_pts, y_plot[i], c=(c,), **scatter_kwargs)

    if annot:
        # Add annotations to each data point, placed on the point directly.
        texts = [ax.text(x_mean[i], y_mean[i], labels[i])
                 for i in range(n_pts)]
        adj_text_kw = {
            'only_move': {'points': 'xy', 'text': 'y', 'objects': 'xy'},
            'arrowprops': {'arrowstyle': '-', 'color': 'gray', 'lw': 0.1}}
        adj_text_kw.update(kwargs.get('adj_text', {}))
        # Iteratively find a better placement for annotations, avoiding overlap.
        adjust_text(texts, x=x_mean, y=y_mean, ax=ax, **adj_text_kw)
    bbox_extra_artists = ()
    if lgd:
        # Place a legend outside the graph, in the top left corner.
        lgd_kwargs = {'bbox_to_anchor': (1.05, 1), 'loc': 2,
                      'borderaxespad': 0, 'ncol': 1}
        lgd_kwargs.update(kwargs.get('lgd', {}))
        lgd = ax.legend(**lgd_kwargs)
        bbox_extra_artists = (lgd,)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.set_tight_layout(tight_layout)
    if save_path:
        fig.savefig(save_path, bbox_extra_artists=bbox_extra_artists,
                    bbox_inches='tight')
    if show:
        fig.show()
    return fig, ax


def cat_errorbar_labelled(y_plot, labels, xlabel=None, ylabel=None,
                          show_yerr=True, cmap='jet', figsize=None, annot=True,
                          save_path=None, show=True, fig=None, ax=None,
                          tight_layout=True, pts_s=6, **kwargs):
    '''
    Errorbar plot using the data `y_plot` and labels `labels` for each point.
    `y_plot` should be a list containing for each point the array of values
    corresponding to each label. The function plots the points with bars going
    from the lowest to the largest value contained in the array, thus showing
    the range of values reached. The labels are added as annotations to the data
    points. The annotations are placed with `adjust_text` to avoid overlapping.
    '''
    scatter_kwargs = kwargs.get('scatter', {'s': 6**2, 'marker': '+'})
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    n_pts = len(labels)
    cmap = plt.cm.get_cmap(cmap, n_pts)
    colors = [cmap(i) for i in range(n_pts)]
    x_plot = np.arange(1, n_pts+1)
    y_mean = [y.mean() for y in y_plot]
    # Sort data's xaxis position by ascending y_mean values.
    sorted_lists = sorted(zip(y_mean, y_plot, labels, colors),
                          key=lambda t: t[0])
    y_mean, y_plot, labels, colors = zip(*sorted_lists)
    yerr = [None] * n_pts
    if show_yerr:
        yerr = [np.array([[np.max(y_mean[i] - y_plot[i])],
                          [np.max(y_plot[i] - y_mean[i])]])
                for i in range(n_pts)]
        x_of_y_pts = [x_plot[i].repeat(len(y_plot[i])) for i in range(n_pts)]
    
    add_objects = []
    for i in range(n_pts):
        c = colors[i]
        plotline, _, barlinecols = ax.errorbar(
            x_plot[i], y_mean[i], yerr=yerr[i], elinewidth=1,
            ls='', marker=None, ms=pts_s, c=c, label=labels[i])
        for b in barlinecols:
            b.set_linestyle(':')
        if show_yerr:
            sub_pts = ax.scatter(x_of_y_pts[i], y_plot[i],
                                 c=(c,), **scatter_kwargs)
            add_objects.append(sub_pts)

    if annot:
        # Add annotations to each errorbar, alternatively on the highest and
        # lowest point.
        texts = []
        for i in range(n_pts):
            y_text = y_plot[i].min() if i % 2 == 0 else y_plot[i].max()
            texts.append(ax.text(x_plot[i], y_text, labels[i]))
        adj_text_kw = {
            'only_move': {'points': 'xy', 'text': 'xy', 'objects': 'xy'},
            'arrowprops': {'arrowstyle': '-', 'color': 'gray', 'lw': 0.1}}
        adj_text_kw.update(kwargs.get('adj_text', {}))
        # Iteratively find a better placement for annotations, avoiding overlap.
        adjust_text(texts, add_objects=add_objects, ax=ax, **adj_text_kw)

    ax.xaxis.set_ticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.set_tight_layout(tight_layout)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        fig.show()
    return fig, ax


def get_ax_height_in(ax, fig):
    '''
    Get the height in inches of an `ax` drawn in a figure `fig`.
    '''
    renderer = fig.canvas.get_renderer()
    disp_to_inch = fig.dpi_scale_trans.inverted()
    return ax.get_tightbbox(renderer).transformed(disp_to_inch).height


def tighth_compos_grids(plot_config, axes_dict, nr_rows=None, nr_cols=None,
                        figsize=None, h_pad=2, save_path=None, show=True):
    '''
    Generates a figure of several rows, where each row's height is unknown and
    cannot be fixed, as the aspect ratio is fixed and we wish to have a certain
    width for the figure (as is usually the case for articles). This is a very
    specific case poorly handled by plt's tight or contrained layouts, as tight
    layout's padding would be different around rows of different heights. Indeed
    since we provide a too large figsize's height, the rows are first spread
    evenly according to their height, a padding different for each row is added
    and it cannot be removed a posteriori. Hence the need to plot each ax a
    first time to get their heights, and then generate the figure again with the
    fitting total height and a gridspec with height_ratios.
    '''
    if nr_rows is None or nr_cols is None:
        nr_rows = len(plot_config)
        nr_cols = 1
    heights = np.zeros(nr_rows)
    for k in range(nr_rows):
        sub_fig, axes = plt.subplots(1, nr_cols, figsize=figsize)
        for i, ax in enumerate(axes.flat):
            i += nr_cols*k
            cell_plot_df = plot_config[i]['cell_plot_df']
            shape_df = plot_config[i]['shape_df']
            sub_fig, ax = grid_viz.plot_grid(cell_plot_df, shape_df, ax=ax,
                                             fig=sub_fig, **axes_dict[i])
            heights[k] = max(get_ax_height_in(ax, sub_fig), heights[k])
        heights[k] = np.max([get_ax_height_in(child, sub_fig)
                             for child in sub_fig.get_children()
                             if isinstance(child, plt.Axes)]
                            + [heights[k]])

    h_pad_in = h_pad*plt.rcParams['font.size']/72
    fig_height_in = sum(heights) + (nr_rows-1) * h_pad_in
    fig, axes = plt.subplots(ncols=nr_cols, nrows=nr_rows,
                             figsize=(figsize[0], fig_height_in),
                             gridspec_kw={'height_ratios': heights})
    for i, ax in enumerate(axes.flat):
        cell_plot_df = plot_config[i]['cell_plot_df']
        shape_df = plot_config[i]['shape_df']
        fig, ax = grid_viz.plot_grid(cell_plot_df, shape_df, ax=ax, fig=fig,
                                     **axes_dict[i])
    fig.subplots_adjust(hspace=h_pad_in / np.mean(heights))
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        fig.show()
    return fig, axes
