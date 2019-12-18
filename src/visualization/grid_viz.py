import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import descartes
import geopandas as geopd

def plot_grid(plot_df, area_df, metric_col='count', save_path=None, show=True,
              title=None, cbar_label=None, **plot_kwargs):
    '''
    Plots the contour of a shape, and on top of it a grid whose cells are
    colored according to the value of a metric for each cell, which are the
    values in the column 'metric_col' of 'plot_df'.
    '''
    fig, ax = plt.subplots(1, figsize=(10, 6))
    # The order here is important, the area's boundaries will be drawn on top of
    # the choropleth.
    plot_df.plot(column=metric_col, ax=ax, **plot_kwargs)
    area_df.plot(ax=ax, color='None', edgecolor='black')
    # plt.colorbar(label='Number of tweets')
    ax.set_title(title)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    if cbar_label:
        vmin = plot_df[metric_col].min()
        vmax = plot_df[metric_col].max()
        # Create colorbar as a legend
        sm = plt.cm.ScalarMappable(cmap=plot_kwargs.get('cmap'),
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # empty array for the data range
        sm._A = []
        # add the colorbar to the figure
        divider = make_axes_locatable(ax)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.1 inch.
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax, label=cbar_label)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    return ax
