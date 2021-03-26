import copy
import IPython.display
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import descartes
import plotly.graph_objects as go
import plotly.colors
import plotly.offline
import numpy as np
import src.utils.scales as scales
import src.utils.geometry as geo

def plot_grid(plot_df, area_df, metric_col='count', save_path=None, show=True,
              title=None, log_scale=False, vmin=None, vmax=None, xy_proj=None,
              cbar_label=None, null_color='k', figsize=None,
              borderwidth=None, cbar_lw=None, ax=None, fig=None, cax=None,
              cbar_ticks=None, annotation=None, show_axes=False,
              tight_layout=True, **kwargs):
    '''
    Plots the contour of a shape, and on top of it a grid whose cells are
    colored according to the value of a metric for each cell, which are the
    values in the column 'metric_col' of 'plot_df'. The figsize provided should
    have the exact wanted value in either the horizontal or vertical direction,
    and one too large in the other one. This function will crop all padding
    in the extra large direction.
    '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    if vmax is None:
        vmax = plot_df[metric_col].max()
    if log_scale:
        if vmin is None:
            vmin = 1
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if vmin is None:
            vmin = plot_df[metric_col].min()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    if xy_proj:
        xlabel = 'position (km)'
        ylabel = 'position (km)'
        plot_df = plot_df.to_crs(xy_proj)
        area_df = area_df.to_crs(xy_proj)
        area_df_bounds = list(area_df.geometry.iloc[0].bounds)
        for i in range(area_df.shape[0]-1):
            new_bounds = area_df.geometry.iloc[i].bounds
            area_df_bounds[0] = min(area_df_bounds[0], new_bounds[0])
            area_df_bounds[1] = min(area_df_bounds[1], new_bounds[1])
        # We translate the whole geometries so that the origin (x,y) = (0,0) is
        # located at the bottom left corner of the shape's bounding box.
        x_off = -area_df_bounds[0]
        y_off = -area_df_bounds[1]
        plot_df.geometry = plot_df.translate(xoff=x_off, yoff=y_off)
        area_df.geometry = area_df.translate(xoff=x_off, yoff=y_off)
    else:
        xlabel = 'longitude (°)'
        ylabel = 'latitude (°)'
    # The order here is important, the area's boundaries will be drawn on top
    # of the choropleth, and the cells with null values will be in null_color.
    area_df.plot(ax=ax, color=null_color, edgecolor='none')
    plot_df.plot(column=metric_col, ax=ax, norm=norm, **kwargs['plot'])
    area_df.plot(ax=ax, color='none', edgecolor='black', linewidth=borderwidth)

    ax.set_title(title)
    if show_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_axis_off()

    if cbar_label:
        cmap = copy.copy(cm.get_cmap(kwargs['plot']['cmap']))
        # Create colorbar as a legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # Bad values, such as a null or negative value for a log scale, are
        # shown in the color set below:
        sm.cmap.set_bad('grey')
        # empty array for the data range
        sm._A = []
        if cax is None:
            # add the colorbar to the figure
            divider = make_axes_locatable(ax)
            # Create an axes on the right side of ax. The width of cax will be
            # 5% of ax and the padding between cax and ax will be fixed at 0.1
            # inch.
            cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(sm, cax=cax, label=cbar_label, ticks=cbar_ticks)
        cbar.solids.set_edgecolor('face')
        cbar.outline.set_lw(cbar_lw)
        cbar = hide_cbar_ext_ticks(cbar, vmin, vmax)
        cbar.ax.tick_params(direction='in', width=0.2, length=2,
                            labelsize=plt.rcParams['font.size']-1, pad=1)


    # Setting the tight layout will make the subplots fit in to the figure area.
    fig.set_tight_layout(tight_layout)
    ax.annotate(annotation, (0, 1), xycoords='axes fraction',
                **kwargs.get('annotate', {}))
    if save_path:
        # However in this case we also need to set bbox_inches to tight because
        # the x/y axes aspect ratio is fixed. So if for instance the figure's
        # x/y ratio is much lower than the x/y aspect ratio of the ax, there
        # would be a lot of vertical padding, which can be removed with
        # bbox_inches='tight'.
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        fig.show()
    return fig, ax


def plot_interactive(raw_cell_plot_df, shape_df, grps_dict, metric_dict,
                     mapbox_style='stamen-toner', mapbox_zoom=6,
                     colorscale=None, plotly_renderer='iframe_connected',
                     save_path=None, show=False, latlon_proj='epsg:4326',
                     alpha=0.8, access_token=None, min_count=1):
    '''
    Plots an interactive Choropleth map with Plotly. The Choropleth data are in
    'cell_plot_df', for each group described in 'grps_dict'.
    Each group's data can be selected with a dropdown menu, which updates
    the figure.
    The map layer on top of which this data is shown is provided by mapbox (see
    https://plot.ly/python/mapbox-layers/#base-maps-in-layoutmapboxstyle for
    possible values of 'mapbox_style').
    Plotly proposes different renderers, described at:
    https://plot.ly/python/renderers/#the-builtin-renderers.
    The geometry column of cell_plot_df must contain only valid geometries:
    just one null value will prevent the choropleth from being plotted.
    '''
    if colorscale is None:
        colorscale = plotly.colors.sequential.Plasma
    cell_plot_df = raw_cell_plot_df.copy()
    start_point = shape_df['geometry'].to_crs(latlon_proj).values[0].centroid
    layout = go.Layout(
        mapbox={'accesstoken': access_token, 'style': mapbox_style,
                'zoom': mapbox_zoom, 'center': {'lat': start_point.y,
                                                'lon': start_point.x}},
        margin={"r": 100, "t": 0, "l": 0, "b": 0})

    # Get a dictionary corresponding to the geojson (because even though the
    # argument is called geojson, it requires a dict type, not a str). The
    # geometry must be in lat, lon.
    geo_dict = cell_plot_df.to_crs(latlon_proj).geometry.__geo_interface__
    # This is the common (and heaviest) part of every choropleth layer as it
    # contains the geometry of the cells.
    choropleth_dict = dict(
        geojson=geo_dict,
        locations=cell_plot_df.index.values,
        hoverinfo='skip',
        marker_opacity=alpha,
        marker_line_width=0.5)

    total_counts = cell_plot_df['total_count'].copy()
    relevance_mask = total_counts < min_count
    total_counts.loc[relevance_mask] = None
    log_counts, count_colorbar = config_log_plot(
        total_counts, 1, total_counts.max())
    count_colorbar['title'] = 'Total count'
    count_colorbar['titleside'] = 'right'
    count_data_dict = {'z': log_counts, 'zmin': 1, 'zmax': log_counts.max(),
                       'colorbar': count_colorbar, 'colorscale': colorscale}
    data = [go.Choroplethmapbox(**choropleth_dict, **count_data_dict)]

    # Since the cells are shared and we simply wish to plot data from different
    # columns of `cell_plot_df`, we create only one Choroplethmapbox. The
    # buttons will then only update the data displayed in every cell and the
    # colorbar. For some reason when we set 'colorbar.tickvals' or
    # 'colorbar.ticktext', these are not updated. So we resort to dtick and
    # pre/suffix. Anything related to the choropleth layer must be encapsulated
    # in a list, anything that is not (like only related to the colorbar's
    # formatting) doesn't need to.
    update_colorscale = [
        [x, color]
        for x, color in zip(np.linspace(0, 1, len(colorscale)), colorscale)]
    but_arg_dict = {'z': [log_counts], 'zmin': [1], 'zmax': [log_counts.max()],
                    'colorscale': [update_colorscale],
                    'colorbar.title': count_colorbar['title'],
                    'colorbar.tickprefix': count_colorbar['tickprefix'],
                    'colorbar.ticksuffix': count_colorbar['ticksuffix'],
                    'colorbar.dtick': 1,
                    }
    buttons = [{'method': 'restyle',
                'args': [but_arg_dict],
                'label': 'Total count'}]

    metric = metric_dict['name']
    readable_metric = metric_dict['readable']
    if metric_dict.get('sym_about') is not None:
        # If the metric requires to be plotted symmetric about a given
        # value, then we use the diverging colorscale RdBu reversed: thus
        # the value corresponding to `sym_about` is white, and the higher
        # values are dark red, and the lower dark blue.
        colorscale = plotly.colors.sequential.RdBu_r

    zmin, zmax = scales.get_global_vmin_vmax(
        cell_plot_df, metric_dict, grps_dict, min_count=min_count)
    og_zmin, og_zmax = zmin, zmax

    for plot_grp in grps_dict:
        grp_dict = grps_dict[plot_grp]
        grp_label = grp_dict['grp_label']
        metric_grp_col = f'{metric}_{plot_grp}'
        if metric_dict.get('log_scale'):
            z, colorbar = config_log_plot(
                cell_plot_df[metric_grp_col], og_zmin, og_zmax)
            zmin = np.log10(og_zmin)
            zmax = np.log10(og_zmax)
        else:
            z = cell_plot_df[metric_grp_col]
            tickvals = np.linspace(zmin, zmax, 6)
            colorbar = {'tickvals': tickvals,
                        'dtick': tickvals[1] - tickvals[0],
                        'tickprefix': '', 'ticksuffix': ''}
        colorbar['title'] = f'{readable_metric} of {grp_label}'
        z.loc[relevance_mask] = None
        update_colorscale = [
            [x, color]
            for x, color in zip(np.linspace(0, 1, len(colorscale)), colorscale)]
        data_dict = {'z': [z], 'zmin': [zmin], 'zmax': [zmax],
                     'colorscale': [update_colorscale],
                     'colorbar.title': colorbar['title'],
                     'colorbar.tickprefix': colorbar['tickprefix'],
                     'colorbar.ticksuffix': colorbar['ticksuffix'],
                     'colorbar.dtick': colorbar['dtick'],
                     }
        buttons.append(dict(
            method='restyle',
            args=[data_dict],
            label=grp_label))

    # Add a dropdown menu to select the data:
    layout.update(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=buttons,
                showactive=True,
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                xanchor="left",
                yanchor="top"
            ),
        ]
    )

    fig = go.Figure(data=data, layout=layout)
    use_iframe_renderer = plotly_renderer.startswith('iframe')
    if save_path:
        plotly.offline.plot(fig, filename=save_path, auto_open=False)
    if show:
        # With the 'iframe' renderer, a standalone HTML is created in a new
        # folder iframe_figures/, and the files are named according to the cell
        # number. Thus, many files can be created while testing out this
        # function, so to avoid this we simply use the previously saved HTML
        # file (so use_iframe_renderer should imply save_path), which has a
        # name we chose.
        if use_iframe_renderer:
            IPython.display.display(IPython.display.IFrame(
                src=save_path, width=900, height=600))
        else:
            fig.show(renderer=plotly_renderer, width=900, height=600)
    return fig, buttons


def config_log_plot(z_series, vmin, vmax):
    '''
    Configures a logarithmic plot from a series `z_series` for a Plotly plot.
    The logarithm in base 10 of the values is calculated, and the ticks'
    positions and labels in accordance. This is for counts only now, so
    values smaller than 1 are not supported.
    '''
    log_counts = z_series.copy()
    log_counts.loc[log_counts <= 0] = None
    log_counts = np.log10(log_counts)
    vmin_log = int(np.log10(vmin))
    vmax_log = int(np.log10(vmax))
    log_ticks = np.arange(vmin_log, vmax_log+1)
    log_colorbar_dict = {'tickvals': log_ticks, 'dtick': 1,
                         'tickprefix': '10<sup>', 'ticksuffix': '<sup>'}
    return log_counts, log_colorbar_dict



def hide_cbar_ext_ticks(cbar, vmin, vmax):
    plt.draw()
    yticks = []
    cbarytks = cbar.ax.get_yticklines()
    # Let's hide the top and bottom most ticks of the colorbar, so that if
    # we set cbar_lw=0, we don't see these, which look weird.
    for i in range(len(cbarytks) // 2):
        y_tick = cbarytks[2*i+1].get_ydata()
        yticks.append(y_tick[0])
        if y_tick[0] in (vmin, vmax) and isinstance(y_tick, tuple):
            cbarytks[2*i+1].set_visible(False)
            cbarytks[2*i].set_visible(False)
    cbar.ax.tick_params(direction='in', width=0.2, length=2,
                        labelsize=plt.rcParams['font.size']-1, pad=1)
    # For some reason the true yticks obtained from get_ticks don't
    # correspond to the ones you get from get_yticklines and get_ydata, so
    # have to force them to the same values to avoid hiding the wrong ticks.
    cbar.set_ticks(yticks)
    return cbar


def get_width_ratios(shape_list, ratio_lgd=0.05, latlon_proj='epsg:4326'):
    '''
    Get the width ratios to pass to a GridSpec so that maps of different regions
    on a same line all fit the full provided height.
    '''
    width_ratios = ratio_lgd * np.ones(len(shape_list) + 1)
    for i, geodf in enumerate(shape_list):
        min_lon, min_lat, max_lon, max_lat = (
            geodf.geometry.to_crs(latlon_proj).total_bounds)
        # For a given longitude extent, the width is maximum the closer to the
        # equator, so the closer the latitude is to 0.
        eq_crossed = min_lat * max_lat < 0
        lat_max_width = min(abs(min_lat), abs(max_lat)) * (1 - int(eq_crossed))
        width = geo.haversine(min_lon, lat_max_width, max_lon, lat_max_width)
        height = geo.haversine(min_lon, min_lat, min_lon, max_lat)
        width_ratios[i] = width / height
    width_ratios[:-1] *= (1 - ratio_lgd) / width_ratios[:-1].sum()
    return width_ratios


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
