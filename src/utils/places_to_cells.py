import pycld2
import geopandas as geopd

LANGS_DICT = dict([(lang[1], lang[0].lower().capitalize())
                   for lang in pycld2.LANGUAGES])

def get_intersect(cells_df, places_geodf, places_counts, xy_proj='epsg:3857'):
    '''
    Get the area of the intersection between the cells in cells_df and the places
    in places_geodf.
    '''
    # We filter out places with a total_count == 0 (ie without residents) via
    # the inner join, so that we don't compute the overlay unnecessarily on
    # these places.
    places_counts_geodf = places_geodf.join(places_counts, how='inner')
    places_counts_geodf['place_id'] = places_counts_geodf.index
    cells_in_places = geopd.overlay(
        places_counts_geodf, cells_df, how='intersection')
    # The geometry of the intersection is projected before calculating the area
    #
    cells_in_places['area_intersect'] = (cells_in_places.geometry
                                                        .to_crs(xy_proj)
                                                        .area)
    return cells_in_places


def get_counts(places_counts, places_langs_counts, places_geodf,
               raw_cells_df, plot_langs_dict, xy_proj='epsg:3857'):
    '''
    From counts by places in 'places_counts' and 'places_langs_counts', returns
    the counts by cells defined in 'cells_df'. 'places_counts' is simply
    a DF with the total and local counts per place. 'places_langs_counts' is a
    multi-index Series with the counts by 'cld_lang' and 'place_id'. The places'
    geometry is defined in 'places_geodf', and the languages we're interested in
    are in 'plot_langs_dict'.
    '''
    cells_df = raw_cells_df.copy()
    # Later index is lost because of the overlay (multiple cells intersect a
    # place), so we duplicate cells_df.index into the 'cell_id' column, to be
    # able to make the groupby in intersect_to_cells
    cells_df['cell_id'] = cells_df.index
    cells_in_places = get_intersect(cells_df, places_geodf, places_counts,
                                    xy_proj=xy_proj)
    cell_plot_df = cells_df.loc[:, ['geometry']]
    cell_plot_df = intersect_to_cells(
        cells_in_places, cell_plot_df, places_counts.columns)

    for plot_lang, lang_dict in plot_langs_dict.items():
        lang_count_col = lang_dict['count_col']
        # We take only the rows where the first level of the index ('cld_lang')
        # is equal to 'plot_lang', and we drop this index level:
        places_lang_counts = (
            places_langs_counts.xs(plot_lang, level='cld_lang')
                               .rename(lang_count_col))
        # We take the same cells_in_places and do a left join here, so we
        # always keep all the rows from the original frame created for the total
        # counts, because having a  non null lang count value implies a non null
        # total count value.
        cells_in_places = cells_in_places.join(
            places_lang_counts, on='place_id', how='left')
        cells_in_places[lang_count_col] = (
            cells_in_places[lang_count_col].fillna(value=0))
        cell_plot_df = intersect_to_cells(
            cells_in_places, cell_plot_df, [lang_count_col])

    return cell_plot_df


def intersect_to_cells(cells_in_places, cells_df, count_cols):
    '''
    Scales all the counts in the columns `count_cols` by the area of
    intersection of the cells with the places, which must have been computed
    prior to calling this function, in the 'area_intersect' column of
    `cells_in_places`. Then the resulting scaled counts are summed by cell.
    '''
    for col in count_cols:
        cells_in_places[col] = cells_in_places[col] * (
            cells_in_places['area_intersect'] / cells_in_places['area'])
        cells_counts = cells_in_places.groupby('cell_id')[col].sum()
        cells_df = cells_df.join(cells_counts, how='left')
    return cells_df
