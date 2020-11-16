import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    inset_axes, Bbox, BboxConnector, BboxPatch, TransformedBbox)
from adjustText import adjust_text
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


def time_evol_ling(ling_props_dict, x_data, idx_data_to_plot=None,
                   idx_ling_to_plot=None, figsize=None, bottom_ylim=0,
                   fig_save_path=None, show=True, color_cycle=None):
    '''
    Plot the time evolution of the proportions of each ling group whose order
    in `ling_props_dict` is contained in `idx_ling_to_plot`. The times are given
    by `x_data`, and the indices of the selected data points are in
    `idx_ling_to_plot`.
    '''
    fig, ax = plt.subplots(1, figsize=figsize)
    ling_labels = list(ling_props_dict.keys())
    if idx_data_to_plot is None:
        idx_data_to_plot = np.arange(0, len(ling_props_dict[ling_labels[0]]))
    if idx_ling_to_plot is None:
        idx_ling_to_plot = range(len(ling_labels))
    if color_cycle is None:
        # This selects the default color cycle.
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x_plot = x_data[idx_data_to_plot]
    for i in idx_ling_to_plot:
        ling_label = ling_labels[i]
        y_plot = np.array(ling_props_dict[ling_label])[idx_data_to_plot]
        # By using the ith element of the color_cycle, we ensure that if we plot
        # multiple graphs, a given language will have a consistent colour.
        ax.scatter(x_plot, y_plot, label=ling_label, c=color_cycle[i])

    if len(ling_props_dict) > 1:
        ax.set_ylim(bottom=bottom_ylim)

    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('proportion')
    fig.set_tight_layout(True)
    if fig_save_path:
        fig.savefig(fig_save_path)
    if show:
        fig.show()
    return ax


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


def scatter_inset(x_data, ling_props_dict, bbox_to_anchor, inset_interval,
                  idx_grp_inset, save_path=None, show=True, figsize=None,
                  ax=None, fig=None, color_cycle=None, top_ylim=None,
                  inset_left=True):
    '''
    Makes a scatter plot of the proportions for each group contained in
    `ling_props_dict` over the times `x_data`, and adds an inset zooming over
    the time interval `inset_interval` of the `idx_grp_inset`-th group. The
    inset is placed within `bbox_to_anchor`.
    '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    if color_cycle is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ling_labels = list(ling_props_dict.keys())
    len_data = len(ling_props_dict[ling_labels[0]])
    idx_data_to_plot = np.arange(0, len_data)
    idx_ling_to_plot = range(len(ling_labels))
    x_plot = x_data[idx_data_to_plot]
    for i in idx_ling_to_plot:
        ling_label = ling_labels[i]
        y_plot = np.array(ling_props_dict[ling_label])[idx_data_to_plot]
        # We plot every step point fixing the maximum number of points to 200,
        # to avoid to have a pdf figure with long loading time.
        step = len_data // 200
        ax.scatter(x_plot[::step], y_plot[::step], label=ling_label,
                   c=color_cycle[i], s=6)

    ax.set_xlabel('t')
    ax.set_ylabel('global proportion')

    y_plot = np.array(ling_props_dict[ling_labels[idx_grp_inset]])[idx_data_to_plot]
    x1, x2 = inset_interval
    y1 = np.min(y_plot[x1:x2])
    y2 = np.max(y_plot[x1:x2])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='30%', pad=0.1)
    # cax.set_axis_off()

    # Place the inset in the Bbox specified in Axes coordinates (from 0 to 1 in
    # ax's size).
    axins = inset_axes(ax, width='100%', height='100%',
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax.transAxes)
    axins.scatter(x_plot[::5], y_plot[::5],
                  label=ling_label, c=color_cycle[idx_grp_inset], s=1)

    # Inset zooms on specific data interval = (x1, x2)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    ax.set_ylim(bottom=0, top=top_ylim)
    # Draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area. As this is a
    # scatter plot and points have a size, it is necessary to enlarge the bbox
    # vertically by an offset to make it visible
    offset = 0.01
    rect = Bbox([[x1, y1-offset], [x2, y2+offset]])
    # A TransformedBbox is initiated with these data coordinates translated to
    # the display coordinate system. This class of Bbox adapts automatically to
    # any potential change made under the hood by plt later on, so that it
    # stays over the specified data range.
    rect = TransformedBbox(rect, ax.transData)
    pp = BboxPatch(rect, fill=False, fc="none", ec="0.5")
    ax.add_patch(pp)

    # If the inset is on the left of the bbox, take the bottom left and top
    # right corners to draw the connectors.
    if inset_left:
        loc11 = 1
        loc12 = 3
    # Else, take the top left and bottom right corners.
    else:
        loc11 = 2
        loc12 = 4
    p1 = BboxConnector(axins.bbox, rect, loc1=loc11, fc="none", ec="0.5")
    axins.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(axins.bbox, rect, loc1=loc12, fc="none", ec="0.5")
    axins.add_patch(p2)
    p2.set_clip_on(False)

    ax.legend()
    fig.set_tight_layout(True)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        fig.show()
    return fig, ax


def scatter_labelled(x_plot, y_plot, labels, xlabel=None, ylabel=None,
                     cmap='jet', figsize=None, annot=True, lgd=None,
                     save_path=None, show=True, **kwargs):
    '''
    Scatter plot using the data (`x_plot`, `y_plot`) and labels `labels` for
    each point. These labels are either assigned to a color in a legend, or
    directly added as annotations to the data points. The annotations are
    placed with `adjust_text` to avoid overlapping.
    '''
    n_pts = len(labels)
    # Sort x_plot, y_plot and labels by ascending x_plot values.
    sorted_lists = sorted(zip(x_plot, y_plot, labels), key=lambda t: t[0])
    x_plot, y_plot, labels = zip(*sorted_lists)
    colors = plt.cm.get_cmap(cmap, n_pts)
    fig, ax = plt.subplots(1, figsize=figsize)
    for i, lab in enumerate(labels):
        ax.scatter(x_plot[i], y_plot[i], label=lab, c=(colors(i),), s=6)

    if annot:
        # Add annotations to each data point, placed on the point directly.
        texts = [ax.text(x_plot[i], y_plot[i], labels[i]) for i in range(n_pts)]
        adj_text_kw = {
            'only_move': {'points': 'xy', 'text': 'y', 'objects': 'xy'},
            'arrowprops': {'arrowstyle': '-', 'color': 'gray', 'lw': 0.1}}
        adj_text_kw.update(kwargs.get('adj_text', {}))
        # Iteratively find a better placement for annotations, avoiding overlap.
        adjust_text(texts, ax=ax, **adj_text_kw)
    if lgd:
        # Place a legend outside the graph, in the top left corner.
        lgd_kwargs = {'bbox_to_anchor': (1.05, 1), 'loc': 2,
                      'borderaxespad': 0, 'ncol': 1}
        lgd_kwargs.update(kwargs.get('lgd', {}))
        lgd = ax.legend(**lgd_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.set_tight_layout(True)
    if save_path:
        fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
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
