import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import descartes
import geopandas as geopd
import IPython.display
import plotly.graph_objects as go
import plotly.offline
import numpy as np

def plot_grid(plot_df, area_df, metric_col='count', save_path=None, show=True,
              title=None, cbar_label=None, log_scale=False, vmax=None,
              xy_proj=None, **plot_kwargs):
    '''
    Plots the contour of a shape, and on top of it a grid whose cells are
    colored according to the value of a metric for each cell, which are the
    values in the column 'metric_col' of 'plot_df'.
    '''
    fig, ax = plt.subplots(1, figsize=(10, 6))
    if vmax is None:
        vmax = plot_df[metric_col].max()
    if log_scale:
        vmin = 1
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = 0
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
    # of the choropleth.
    plot_df.plot(column=metric_col, ax=ax, norm=norm, **plot_kwargs)
    area_df.plot(ax=ax, color='None', edgecolor='black')

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


def plot_interactive(raw_cell_plot_df, shape_df, plot_langs_dict,
                     mapbox_style='stamen-toner', mapbox_zoom=6,
                     colorscale='Plasma', plotly_renderer=None,
                     save_path=None, show=False):
    '''
    Plots an interactive Choropleth map with Plotly. The Choropleth data are in
    'cell_plot_df', for each language described in 'plot_langs_dict'.
    Each language's data can be selected with a dropdown menu, which updates
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
    start_point = shape_df['geometry'].values[0].centroid
    layout = go.Layout(
        mapbox_style=mapbox_style, mapbox_zoom=mapbox_zoom,
        mapbox_center={"lat": start_point.y, "lon": start_point.x},
        margin={"r": 100, "t": 0, "l": 0, "b": 0})

    # Get a dictionary corresponding to the geojson (because even though the
    # argument is called geojson, it requires a dict type, not a str). The
    # geometry must be in lat, lon
    geo_dict = cell_plot_df.geometry.__geo_interface__
    choropleth_dict = dict(
        geojson=geo_dict,
        locations=cell_plot_df.index.values,
        hoverinfo='skip',
        colorscale=colorscale,
        marker_opacity=0.5, marker_line_width=0)

    log_count = cell_plot_df['total_count'].copy()
    log_count.loc[log_count < 1] = None
    log_count = np.log10(log_count)
    log_ticks = np.arange(0, log_count.max()+1)
    log_ticks_text = ['10<sup>{:n}</sup>'.format(tick) for tick in log_ticks]
    total_count_colorbar = {'title':'Total count', 'titleside':'right',
                            'tickvals': log_ticks, 'ticktext': log_ticks_text}
    data = [go.Choroplethmapbox(**choropleth_dict,
                                z=log_count,
                                colorbar=total_count_colorbar,
                                visible=True)]

    nr_layers = len(plot_langs_dict) + 1
    # Each button in the dropdown menu corresponds to a state, where one
    # choropleth layer is visible and all others are hidden. The order in the
    # list in 'visible' corresponds to the order of the list data.
    buttons = [dict(
        method='restyle',
        args=[{'visible': [True] + [False]*(nr_layers-1)}],
        label='Total count')]

    for i, plot_lang in enumerate(plot_langs_dict):
        lang_dict = plot_langs_dict[plot_lang]
        readable_lang = lang_dict['readable']
        prop_lang_col = lang_dict['prop_col']
        cbar_label = lang_dict['prop_label']
        data.append(go.Choroplethmapbox(**choropleth_dict,
                                        z=cell_plot_df[prop_lang_col],
                                        zmin=0, zmax=1,
                                        colorbar={'title':cbar_label,
                                                  'titleside':'right'},
                                        visible=False))
        visible = [False] * nr_layers
        visible[i+1] = True
        buttons.append(dict(
            method='restyle',
            args=[{'visible': visible}],
            label=readable_lang))

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
