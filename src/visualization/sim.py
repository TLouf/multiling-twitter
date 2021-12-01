import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import (
    inset_axes, Bbox, BboxConnector, BboxPatch, TransformedBbox)
import src.visualization.grid_viz as grid_viz

def time_evol_ling(ling_props_dict, x_data, idx_data_to_plot=None,
                   idx_ling_to_plot=None, figsize=None, bottom_ylim=0, ax=None,
                   fig_save_path=None, show=True, color_cycle=None, legend=True,
                   **scatter_kwargs):
    '''
    Plot the time evolution of the proportions of each ling group whose order
    in `ling_props_dict` is contained in `idx_ling_to_plot`. The times are given
    by `x_data`, and the indices of the selected data points are in
    `idx_ling_to_plot`.
    '''
    if ax is None:
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
        ax.scatter(x_plot, y_plot, label=ling_label, c=(color_cycle[i],),
                   **scatter_kwargs)

    if len(ling_props_dict) > 1:
        ax.set_ylim(bottom=bottom_ylim)

    if legend:
        ax.legend()
    ax.set_xlabel('step')
    ax.set_ylabel(r'$p_{L}$')
    if fig_save_path:
        fig.savefig(fig_save_path)
    if show:
        fig.show()
    return ax


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
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        fig.show()
    return fig, ax


def phase_space(ax, converg_dict, list_s, list_q, colors, rasterized=False):
    q_grid, s_grid = np.meshgrid(list_q, list_s)
    color_mat = np.empty(s_grid.shape)
    color_mat[:] = np.nan
    for k, list_sq in enumerate(converg_dict.values()):
        for (s, q) in list_sq:
            i = np.where(list_s == s)[0][0]
            j = np.where(list_q == q)[0][0]
            color_mat[i, j] = k
    # Forward fill possible missing values due to irregular grid
    mask = np.isnan(color_mat)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    color_mat[mask] = color_mat[np.nonzero(mask)[0], idx[mask]]

    my_cmap = LinearSegmentedColormap.from_list('mine', colors)
    norm = plt.Normalize(vmin=0, vmax=len(colors)-1)
    pcol = ax.pcolormesh(s_grid, q_grid, color_mat, shading='nearest',
                         cmap=my_cmap, norm=norm, linewidth=0,
                         rasterized=rasterized)
    pcol.set_edgecolor('face')

    for converg_kind, sq in converg_dict.items():
        if len(sq) > 0:
            ax.scatter(*zip(*sq), label=converg_kind, c='w', alpha=0.5, s=0)

    ax.autoscale(False)
    guide_c = np.array([255]*3) / 255
    ax.plot([0, 1], [1, 0], ls='--', c=guide_c, lw=0.5)

    ax.set_xlim(list_s[0], list_s[-1])
    ax.set_ylim(list_q[0], list_q[-1])
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$q$')
    ax.set_aspect('equal')
    return ax


def phase_space_eg(converg_dict, list_s, list_q, plot_params, c_values,
                   cell_plot_df, shape_df, fig, ax, axins, colors,
                   tight_layout=True, cax=None, rasterized=False):
    c = plot_params['c']
    r = plot_params['mu'] / (c * (1-plot_params['mu']))
    s_case = plot_params['s']
    q_case = plot_params['q']

    ax = phase_space(ax, converg_dict, list_s, list_q, colors,
                     rasterized=rasterized)
    ax.set_title(f'$c = {c}$ ($r = {r:.2g}$)')
    if c != c_values[0]:
        ax.set_ylabel(None)
        ax.tick_params(left=False, labelleft=False)

    cmap = LinearSegmentedColormap.from_list('my_cmap', colors[[0, 1, 3]])
    plot_kwargs = {'edgecolor': (0.9, 0.9, 0.9), 'linewidths': 0.1,
                   'cmap': cmap, 'rasterized': rasterized}
    if c == c_values[-1]:
        cbar_label = r'French polarization'
    else:
        cbar_label = None
    fig, axins = grid_viz.plot_grid(
        cell_plot_df, shape_df, metric_col='fr_polar', ax=axins, fig=fig,
        cbar_label=cbar_label, cbar_lw=0, vmin=0, vmax=1, cax=cax,
        show_axes=True, borderwidth=0.2, tight_layout=tight_layout,
        **{'plot': plot_kwargs})
    axins.tick_params(bottom=False, labelbottom=False,
                      left=False, labelleft=False)
    axins.set_xlabel(None)
    axins.set_ylabel(None)

    offset = 0.005
    rect = Bbox([[s_case-offset, q_case-offset],
                 [s_case+offset, q_case+offset]])
    rect = TransformedBbox(rect, ax.transData)
    pp = BboxPatch(rect, fill=False, fc="none", ec="0.3")
    ax.add_patch(pp)
    return fig, ax, axins


def scatter_interp(fig, ax, x_plot, y_plot, t_plot, stable_idc, arrow_steps,
                   **scatter_kwargs):
    nr_interp_pts = len(t_plot) * 10
    interp_t = np.linspace(0, 1, nr_interp_pts)
    tck, _ = interpolate.splprep([x_plot, y_plot], s=0)
    xi, yi = interpolate.splev(interp_t, tck)
    vmax = t_plot[-1]

    plasma_cmap = plt.cm.get_cmap('viridis', vmax)
    for i in range(nr_interp_pts):
        _, = ax.plot(xi[i:i+2], yi[i:i+2], ls=':', lw=0.5,
                     c=plasma_cmap(interp_t[i]))

    arrow_idc = np.searchsorted(interp_t, arrow_steps/vmax)
    for start_idx in arrow_idc:
        c = plasma_cmap(interp_t[start_idx])
        ax.annotate(
            '',
            xytext=(xi[start_idx], yi[start_idx]),
            xy=(xi[start_idx + 1], yi[start_idx + 1]),
            arrowprops={'arrowstyle': '->', 'color': c, 'lw': 1}, size=8)

    for (t, x, y) in zip(t_plot[stable_idc],
                         x_plot[stable_idc],
                         y_plot[stable_idc]):
        ax.scatter(x, y, label=str(t), c=(plasma_cmap(t),), zorder=2.5,
                   **scatter_kwargs)
    ax.legend(title='step', handletextpad=0.5, labelspacing=0.3,
              fontsize=plt.rcParams['font.size']-1, loc='center left')

    remain_idc = [i for i in range(len(t_plot)) if i not in stable_idc]
    for (t, x, y) in zip(t_plot[remain_idc],
                         x_plot[remain_idc],
                         y_plot[remain_idc]):
        ax.scatter(x, y, c='k', marker='.', zorder=2, s=2)


    return fig, ax
