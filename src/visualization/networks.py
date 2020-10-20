import numpy as np
import matplotlib.collections as collections
import matplotlib.pyplot as plt
import geopandas as geopd
import networkx as nx


def edge_dist_distrib(cell_interactions, cell_centers, ylabel, nr_bins=20,
                      save_path=None, show=True):
    '''
    Plots the distribution of interactions in function of the distance between
    their cell of origin and destination.
    '''
    dist_distrib = (
        cell_interactions.join(cell_centers, on='cell_id')
                         .join(cell_centers.rename('to_cell_center'),
                               on='to_cell_id'))
    dist_distrib['dist'] = (
        geopd.GeoSeries(dist_distrib[cell_centers.name]).distance(
            geopd.GeoSeries(dist_distrib['to_cell_center'])) / 1000).astype(int)
    max_dist = dist_distrib['dist'].max()
    # +1 to include the max in the last bin
    bin_size = (max_dist+1) / nr_bins
    bins = np.arange(0, max_dist, bin_size)
    dist_distrib['bin'] = dist_distrib['dist'] // bin_size
    bin_heights = dist_distrib.groupby('bin')['whole_ratio'].sum()
    bin_heights = bin_heights.reindex(range(nr_bins), fill_value=0)
    plt.xlabel('distance (km)')
    plt.ylabel(ylabel)
    plt.bar(bins, bin_heights, width=bin_size)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_cell_interactions(cell_interactions, cells_in_area_df, shape_df, title=None,
                           edge_cmap='plasma', edge_th=10, save_path=None,
                           show=True, cbar_label=None):
    '''
    Plots the interactions between cells, coloring edges from one cell to
    another according to the number of users interacting in a given language,
    given in `cell_interactions`. The network is directed, from sender to
    receiver. The nodes, at the center of each cell, are represented with a size
    increasing with the strength of the self loop. All geometries in
    `cells_in_area_df` and `shape_df` are assumed to be in the same xy
    projection.
    '''
    area_df = shape_df.copy()
    cells_df = cells_in_area_df.copy()
    area_df_bounds = area_df.geometry.iloc[0].bounds
    # We translate the whole geometries so that the origin (x,y) = (0,0) is
    # located at the bottom left corner of the shape's bounding box.
    x_off = -area_df_bounds[0]
    y_off = -area_df_bounds[1]
    cells_df.geometry = cells_df.translate(xoff=x_off, yoff=y_off)
    area_df.geometry = area_df.translate(xoff=x_off, yoff=y_off)
    
    is_self_loop = cell_interactions['cell_id'] == cell_interactions['to_cell_id']
    self_interactions = (
        cell_interactions.loc[is_self_loop]
                         .set_index(['cell_id', 'to_cell_id'])['whole_ratio']
                         .groupby('cell_id')
                         .sum())
    print(f'self: {self_interactions.sum()}')
    data = (cell_interactions.loc[~is_self_loop]
                             .set_index(['cell_id', 'to_cell_id'])['whole_ratio']
                             .copy())
    print(f'not self: {data.sum()}')
    # To avoid getting too cluttered a plot, we only keep the edges with a
    # sufficient weight, and then we sort the edges by weight in order to draw
    # the highest weighted nodes on top, so that they don't get hidden by less
    # important edges
    data = data.loc[data > edge_th].sort_values()
    print(f'not self, above threshold: {data.sum()}')
    node_list = np.unique(
        data.index.get_level_values('to_cell_id').unique().tolist()
        + data.index.get_level_values('cell_id').unique().tolist()).tolist()
    node_sizes = self_interactions
    node_sizes = node_sizes.reindex(node_list).fillna(0)
    edge_list = list(data.keys())
    G = nx.MultiDiGraph(edge_list)
    pos = {cell_id: np.array((pt.x, pt.y))
           for cell_id, pt in cells_df.geometry.centroid.iteritems()}
    # max size is the cell size, and the order on node_sizes must be the same as
    # in nodelist (which is G.nodes by default). node_sizes is in a weird unit,
    # but seems to be an area, not a radius
    
    cell_size = cells_in_area_df.cell_size
    node_sizes = 3 + cell_size/100 * ((node_sizes.values - node_sizes.min())
                                      / (node_sizes.max() - node_sizes.min()))
    edge_colors = data.values
    edge_alphas = 0.3 + 0.7 * ((data.values - data.values.min())
                               / (data.values.max() - data.values.min()))
    _, ax = plt.subplots(1, figsize=(10, 6))
    # We draw in the background the shape of the region and the grid lines
    cells_df.plot(ax=ax, color='None', edgecolor='grey', linewidths=0.2)
    area_df.plot(ax=ax, color='None', edgecolor='k')
    
    _ = nx.draw_networkx_nodes(G, pos,
                                   nodelist=node_list, node_size=node_sizes,
                                   node_color="blue", ax=ax)
    # For some reason, G.edges has a different order than the list it uses to
    # draw the graph from, so we need to re specify the edgelist here, to make
    # sure the correct weight is associated to the correct edge.
    edges = nx.draw_networkx_edges(
        G, pos, node_size=node_sizes, nodelist=node_list, arrowstyle="->",
        arrowsize=4, edgelist=edge_list, edge_color=edge_colors,
        edge_cmap=edge_cmap, width=1, connectionstyle='arc3, rad = 0.1', ax=ax)
    # set alpha value for each edge
    for i, alpha in enumerate(edge_alphas):
        edges[i].set_alpha(alpha)

    if cbar_label:
        # All the following is just needed to draw the colorbar, the collection
        # is not added to the plot
        pc = collections.PatchCollection(edges, cmap=edge_cmap)
        pc.set_array(edge_colors)
        plt.colorbar(pc, label=cbar_label)

    # nx disables xticks, so we reset the tick params back to defaults
    ax.tick_params(reset=True)
    xticks_km = ax.get_xticks() / 1000
    ax.set_xticklabels([f'{t:.0f}' for t in xticks_km])
    yticks_km = ax.get_yticks() / 1000
    ax.set_yticklabels([f'{t:.0f}' for t in yticks_km])
    plt.xlabel('position (km)')
    plt.ylabel('position (km)')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return ax
