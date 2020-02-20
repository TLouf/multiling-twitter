import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import descartes
import geopandas as geopd
import IPython.display
import plotly.graph_objects as go
import plotly.offline
import numpy as np
import src.utils.scales as scales

def plot_grid(plot_df, area_df, metric_col='count', save_path=None, show=True,
              title=None, log_scale=False, vmin=None, vmax=None, xy_proj=None,
              cbar_label=None, null_color='None', **plot_kwargs):
    '''
    Plots the contour of a shape, and on top of it a grid whose cells are
    colored according to the value of a metric for each cell, which are the
    values in the column 'metric_col' of 'plot_df'.
    '''
    fig, ax = plt.subplots(1, figsize=(10, 6))
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
        area_df_bounds = area_df.geometry.iloc[0].bounds
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
    # of the choropleth, and the cells with null values will be in null_color
    area_df.plot(ax=ax, color=null_color, edgecolor='none')
    plot_df.plot(column=metric_col, ax=ax, norm=norm, **plot_kwargs)
    area_df.plot(ax=ax, color='none', edgecolor='black')

    if xy_proj:
        xticks_km = ax.get_xticks() / 1000
        ax.set_xticklabels([f'{t:.0f}' for t in xticks_km])
        yticks_km = ax.get_yticks() / 1000
        ax.set_yticklabels([f'{t:.0f}' for t in yticks_km])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_title(title)
    if cbar_label:
        # Create colorbar as a legend
        sm = plt.cm.ScalarMappable(cmap=plot_kwargs.get('cmap'), norm=norm)
        # Bad values, such as a null or negative value for a log scale, are
        # shown in the color set below:
        sm.cmap.set_bad('grey')
        # empty array for the data range
        sm._A = []
        # add the colorbar to the figure
        divider = make_axes_locatable(ax)
        # Create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.1 inch.
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax, label=cbar_label)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    return ax


def plot_interactive(raw_cell_plot_df, shape_df, grps_dict, metric_dict,
                     mapbox_style='stamen-toner', mapbox_zoom=6,
                     colorscale='Plasma', plotly_renderer='iframe_connected',
                     save_path=None, show=False, latlon_proj='epsg:4326'):
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
    cell_plot_df = raw_cell_plot_df.copy()
    start_point = shape_df['geometry'].to_crs(latlon_proj).values[0].centroid
    layout = go.Layout(
        mapbox_style=mapbox_style, mapbox_zoom=mapbox_zoom,
        mapbox_center={"lat": start_point.y, "lon": start_point.x},
        margin={"r": 100, "t": 0, "l": 0, "b": 0})

    # Get a dictionary corresponding to the geojson (because even though the
    # argument is called geojson, it requires a dict type, not a str). The
    # geometry must be in lat, lon.
    geo_dict = cell_plot_df.to_crs(latlon_proj).geometry.__geo_interface__
    choropleth_dict = dict(
        geojson=geo_dict,
        locations=cell_plot_df.index.values,
        hoverinfo='skip',
        colorscale=colorscale,
        marker_opacity=0.8, marker_line_width=0)

    total_counts = cell_plot_df['total_count'].copy()
    total_counts.loc[total_counts < 1] = None
    log_counts, count_colorbar = config_log_plot(
        total_counts, 1, total_counts.max())
    count_colorbar['title'] = 'Total count'
    count_colorbar['titleside'] = 'right'
    data = [go.Choroplethmapbox(**choropleth_dict,
                                z=log_counts,
                                colorbar=count_colorbar,
                                visible=True)]

    nr_layers = len(grps_dict) + 1
    # Each button in the dropdown menu corresponds to a state, where one
    # choropleth layer is visible and all others are hidden. The order in the
    # list in 'visible' corresponds to the order of the list `data`.
    buttons = [dict(
        method='restyle',
        args=[{'visible': [True] + [False]*(nr_layers-1)}],
        label='Total count')]

    metric = metric_dict['name']
    if metric_dict.get('sym_about') is not None:
        # If the metric requires to be plotted symmetric about a given
        # value, then we use the diverging colorscale RdBu reversed: thus
        # the value corresponding to `sym_about` is white, and the higher
        # values are dark red, and the lower dark blue.
        choropleth_dict['colorscale'] = 'RdBu'
        choropleth_dict['reversescale'] = True

    zmin, zmax = scales.get_global_vmin_vmax(
        cell_plot_df, metric_dict, grps_dict, min_count=0)
    og_zmin, og_zmax = zmin, zmax

    for i, plot_grp in enumerate(grps_dict):
        grp_dict = grps_dict[plot_grp]
        readable_grp = grp_dict['readable']
        metric_grp_col = grp_dict[metric + '_col']
        if metric_dict.get('log_scale'):
            z, colorbar = config_log_plot(
                cell_plot_df[metric_grp_col], og_zmin, og_zmax)
            zmin = np.log10(og_zmin)
            zmax = np.log10(og_zmax)
        else:
            z = cell_plot_df[metric_grp_col]
            colorbar = {}
        colorbar['title'] = grp_dict[metric + '_label']
        colorbar['titleside'] = 'right'

        data.append(go.Choroplethmapbox(**choropleth_dict,
                                        z=z,
                                        zmin=zmin,
                                        zmax=zmax,
                                        colorbar=colorbar,
                                        visible=False))
        visible = [False] * nr_layers
        visible[i+1] = True
        buttons.append(dict(
            method='restyle',
            args=[{'visible': visible}],
            label=readable_grp))

    # Add a dropdown menu to select the data:
    layout.update(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
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
            fig.show(renderer=plotly_renderer, width=900, height=600,
                     config={'modeBarButtonsToAdd': ['zoomInMapbox', 'zoomOutMapbox']})

    return fig


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
    log_ticks_text = ['10<sup>{:n}</sup>'.format(tick) for tick in log_ticks]
    log_colorbar_dict = {'tickvals': log_ticks, 'ticktext': log_ticks_text}
    return log_counts, log_colorbar_dict
