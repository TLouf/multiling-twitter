import pycld2
import geopandas as geopd
import src.data.user_agg as uagg

LANGS_DICT = dict([(lang[1], lang[0].lower().capitalize())
                   for lang in pycld2.LANGUAGES])

def get_intersect(cells_df, places_geodf, places_counts):
    '''
    Get the area of the intersection between the cells in cells_df and the
    places in places_geodf.
    '''
    # We filter out places with a total_count == 0 (ie without residents) via
    # the inner join, so that we don't compute the overlay unnecessarily on
    # these places.
    places_counts_geodf = places_geodf.join(places_counts, how='inner')
    places_counts_geodf['place_id'] = places_counts_geodf.index
    cells_in_places = geopd.overlay(
        places_counts_geodf, cells_df, how='intersection')
    cells_in_places['area_intersect'] = cells_in_places.geometry.area
    return cells_in_places


def get_counts(places_counts, places_langs_counts, places_geodf,
               raw_cells_df, plot_langs_dict):
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
    cells_in_places = get_intersect(cells_df, places_geodf, places_counts)
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


def get_all_area_counts(users_home_areas, user_langs_agg, users_ling_grp,
                        plot_langs_dict, plot_lings_dict):
    '''
    Based on which language group the users belong to (given in `user_langs_agg`
    and `users_ling_grp`) and in which area they reside (`users_home_areas`), 
    gets the counts of users belonging to every group in every area.
    '''
    # Initialize with total counts
    areas_counts = (users_home_areas.to_frame()
                                    .groupby(users_home_areas.name)
                                    .size()
                                    .rename('total_count')
                                    .to_frame())
    # We count the number of users speaking a local language in each cell and
    # place of residence.
    local_lang_users = user_langs_agg.reset_index(level='cld_lang')
    local_langs = [lang for lang in plot_langs_dict]
    local_langs_mask = local_lang_users['cld_lang'].isin(local_langs)
    local_lang_users = (local_lang_users.loc[local_langs_mask]
                                        .groupby('uid')
                                        .first())
    areas_local_counts = uagg.to_count_by_area(
        local_lang_users, users_home_areas, output_col='local_count')
    # Then we get the counts of speakers by language and cell
    areas_langs_counts = uagg.to_count_by_area(user_langs_agg, users_home_areas)
    # Then the counts of groups (mono-, bi-, tri-linguals):
    areas_ling_counts = uagg.to_count_by_area(users_ling_grp, users_home_areas)
    # We always left join on places counts, because total_count == 0 implies
    # that every other count is 0.
    areas_counts = areas_counts.join(areas_local_counts, how='left')
    existing_lings = areas_ling_counts.index.get_level_values('ling_grp')
    for ling, ling_dict in plot_lings_dict.items():
        ling_count_col = ling_dict['count_col']
        if ling in existing_lings:
            areas_grp_count = (areas_ling_counts.xs(ling, level='ling_grp')
                                                .rename(ling_count_col))
            areas_counts = areas_counts.join(areas_grp_count, how='left')
        else:
            areas_counts[ling_count_col] = None

    existing_langs = areas_langs_counts.index.get_level_values('cld_lang')
    for lang, lang_dict in plot_langs_dict.items():
        lang_count_col = lang_dict['count_col']
        if lang in existing_langs:
            areas_lang_counts = (areas_langs_counts.xs(lang, level='cld_lang')
                                                   .rename(lang_count_col))
            areas_counts = areas_counts.join(areas_lang_counts, how='left')
        else:
            areas_counts[lang_count_col] = None

    return areas_counts


def home_places_to_cells(cell_plot_df, user_only_place, places_geodf,
                         user_langs_agg, users_ling_grp, plot_langs_dict,
                         plot_lings_dict):
    '''
    Appends new columns to `cell_plot_df` with the counts by cell of the
    language groups described in `plot_langs_dict` and `plot_lings_dict`. The
    counts are calculated from the user counts in `user_langs_agg` and
    `users_ling_grp` respectively, for the users who only have a place of
    residence, which is given in `user_only_place`. The cell counts are then
    distributed proportionally to the area of the place which intersects the
    cell(s).
    '''
    places_counts = get_all_area_counts(
        user_only_place, user_langs_agg, users_ling_grp, plot_langs_dict,
        plot_lings_dict)
    cells_in_places = get_intersect(cell_plot_df, places_geodf, places_counts)
    count_cols = places_counts.columns
    cell_plot_df = intersect_to_cells(cells_in_places, cell_plot_df, count_cols)
    return cell_plot_df
