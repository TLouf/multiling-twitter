import matplotlib.cm as cm
import matplotlib.pyplot as plt
import descartes
import geopandas as geopd

def plot_grid(cells_df, cell_tweet_counts_df, area_df, cmap='Blues'):
    '''
    Plots the contour of a shape, and on top of it a grid whose cells are
    colored according to the value of a metric for each cell.
    '''
    plot_df = cells_df.join(cell_tweet_counts_df, how='inner')
    my_cmap = cm.get_cmap(cmap)
    plt.figure()
    ax = plt.gca()
    plot_df.plot(column='count', ax=ax, legend=True, edgecolor='w', cmap=my_cmap)
    area_df.plot(ax=ax, color='None', edgecolor='black')
    plt.show()
    plt.clf()
