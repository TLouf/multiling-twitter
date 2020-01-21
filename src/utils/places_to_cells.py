import pycld2
import geopandas as geopd

LANGS_DICT = dict([(lang[1], lang[0].lower().capitalize())
                   for lang in pycld2.LANGUAGES])

def get_counts(places_counts, places_langs_counts, places_geodf,
               raw_cells_df, plot_langs_dict, xy_proj='epsg:3857'):
    '''
    From counts by places in 'places_counts' and 'places_langs_counts', returns
    the counts by cells defined in 'cells_df'. 'places_counts' is simply
    a Series with the total counts per place. 'places_langs_counts' is a
    multi-index Series with the counts by 'cld_lang' and 'place_id'. The places'
    geometry is defined in 'places_geodf', and the languages we're interested in
    are in 'plot_langs_dict'.
    '''
    cells_df = raw_cells_df.copy()
    cells_df['cell_id'] = cells_df.index
    places_counts_geodf = places_geodf.join(places_counts, how='inner')
    places_counts_geodf['place_id'] = places_counts_geodf.index
    cells_in_places = geopd.overlay(
        places_counts_geodf, cells_df, how='intersection')
    cells_in_places['area_intersect'] = (cells_in_places.geometry
                                                        .to_crs(xy_proj)
                                                        .area)
    cells_in_places['total_count'] = cells_in_places['count'] * (
            cells_in_places['area_intersect'] / cells_in_places['area'])
    cells_in_places = cells_in_places.drop(columns=['count'])
    # Here the index is lost because of the overlay (multiple cells intersect a
    # place), that's why  we had to duplicate cells_df.index into the
    # 'cell_id' column, to be able to make the groupby
    total_cells_counts = cells_in_places.groupby('cell_id')['total_count'].sum()
    cell_plot_df = (cells_df.loc[:, ['geometry']]
                            .join(total_cells_counts, how='left'))

    for plot_lang, lang_dict in plot_langs_dict.items():
        lang_count_col = lang_dict['count_col']
        # We take only the rows where the first level of the index ('cld_lang')
        # is equal to 'plot_lang', and we drop this index level:
        places_lang_counts = places_langs_counts.xs(plot_lang, level='cld_lang')
        # We take the same cells_in_places and do a left join here, so we
        # always keep all the rows from the original frame created for the total
        # counts, because having a  non null lang count value implies a non null
        # total count value.
        cells_in_places = cells_in_places.join(
            places_lang_counts, on='place_id', how='left')
        cells_in_places['count'] = cells_in_places['count'].fillna(value=0)
        cells_in_places[lang_count_col] = cells_in_places['count'] * (
            cells_in_places['area_intersect'] / cells_in_places['area'])
        cells_in_places = cells_in_places.drop(columns=['count'])

        lang_cells_counts = (cells_in_places.groupby('cell_id')[lang_count_col]
                                            .sum())
        cell_plot_df = cell_plot_df.join(lang_cells_counts, how='left')

    return cell_plot_df
