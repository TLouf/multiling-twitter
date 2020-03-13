import os
import multiprocessing as mp
import pandas as pd
import geopandas as geopd
import src.data.user_agg as uagg
import src.data.access as data_access
import src.utils.places_to_cells as places_to_cells
import src.utils.join_and_count as join_and_count
import src.utils.geometry as geo
import src.utils.make_config as make_config


def from_users_area_and_lang(cells_in_area_df, places_geodf, user_only_place,
                             user_home_cell, user_langs_agg, users_ling_grp,
                             plot_langs_dict, multiling_grps, 
                             cell_data_path_format):
    '''
    Generates `cell_plot_df` assuming we already have the language and residence
    attribution done of all users (so for a given cell size).
    '''
    cc = places_geodf.cc
    cell_size = cells_in_area_df.cell_size
    cell_plot_df = cells_in_area_df.copy()
    cell_plot_df = places_to_cells.home_places_to_cells(
        cell_plot_df, user_only_place, places_geodf,
        user_langs_agg, users_ling_grp, plot_langs_dict, multiling_grps)
    # Then we add the total counts for the users with a cell of residence.
    cells_counts = places_to_cells.get_all_area_counts(
        user_home_cell, user_langs_agg, users_ling_grp, plot_langs_dict,
        multiling_grps)
    count_cols = cells_counts.columns
    for col in count_cols:
        cell_plot_df = join_and_count.increment_join(
            cell_plot_df, cells_counts[col], count_col=col)
    cell_plot_df = cell_plot_df.loc[cell_plot_df['total_count'] > 0]

    for plot_lang, lang_dict in plot_langs_dict.items():
        lang_count_col = lang_dict['count_col']
        level_lang_label = lang_dict['grp_label']
        sum_lang = cell_plot_df[lang_count_col].sum()
        print(f'There are {sum_lang:.0f} {level_lang_label}.')

    cell_data_path = cell_data_path_format.format('users', cc, cell_size)
    print(f'saving at {cell_data_path}.')
    cell_plot_df.to_file(cell_data_path, driver='GeoJSON')
    return cell_plot_df


def from_uagg_res(user_agg_res, cells_df_list, places_geodf, plot_langs_dict,
                  area_dict, cell_data_path_format, lang_relevant_prop=0.1,
                  lang_relevant_count=5, cell_relevant_th=0.1,
                  place_relevant_th=0.1, fig_dir=None):
    '''
    Generates `cell_plot_df` assuming we have the results of
    `user_agg.get_lang_loc_habits`, which don't depend on the cell size, so we
    can get multiple `cell_plot_df` for the cell sizes in `cells_df_list`.
    '''
    user_langs_counts = join_and_count.init_counts(['uid', 'cld_lang'])
    user_places_habits = join_and_count.init_counts(['uid', 'place_id',
                                                     'isin_workhour'])
    for lang_res, cell_res, place_res in user_agg_res:
        user_langs_counts = join_and_count.increment_join(user_langs_counts,
                                                          lang_res)
        user_places_habits = join_and_count.increment_join(user_places_habits,
                                                           place_res)

    # We first do the language attribution for all users.
    user_langs_agg = uagg.get_lang_grp(user_langs_counts, area_dict,
                                       lang_relevant_prop=lang_relevant_prop,
                                       lang_relevant_count=lang_relevant_count,
                                       fig_dir=fig_dir)
    users_ling_grp, multiling_grps = uagg.get_ling_grp(
        user_langs_agg, area_dict,
        lang_relevant_prop=lang_relevant_prop,
        lang_relevant_count=lang_relevant_count, fig_dir=fig_dir)
    print('lang attribution done')
    # Now we iterate over all cell sizes
    for i, cells_df in enumerate(cells_df_list):
        user_cells_habits = join_and_count.init_counts(['uid', 'cell_id',
                                                        'isin_workhour'])
        # We first construct user_cells_habits with the corresponding result
        # from user_agg_res.
        for _, cell_res, _ in user_agg_res:
            user_cells_habits = join_and_count.increment_join(
                user_cells_habits, cell_res[i])

        # And then we can get the residence of each user, and can get
        # cell_plot_df for this cell size, which we save on disk.
        user_home_cell, user_only_place = uagg.get_residence(
            user_cells_habits, user_places_habits,
            place_relevant_th=place_relevant_th,
            cell_relevant_th=cell_relevant_th)
        cell_plot_df = from_users_area_and_lang(
            cells_df, places_geodf, user_only_place,
            user_home_cell, user_langs_agg, users_ling_grp,
            plot_langs_dict, multiling_grps, cell_data_path_format)


def from_scratch(area_dict, valid_uids, places_geodf, cells_df_list,
                 tweets_files_paths, get_df_fun, collect_user_agg_res, 
                 user_agg_res, langs_agg_dict, cell_data_path_format, 
                 lang_relevant_prop=0.1, lang_relevant_count=5, 
                 cell_relevant_th=0.1, place_relevant_th=0.1, fig_dir=None,
                 cpus=8):
    '''
    Generates `cell_plot_df` only assuming the list of valid users has already
    been generated thanks to the filters in the `user_filters` module.
    '''
    user_level_label = '{}-speaking users'
    plot_langs_dict = make_config.langs_dict(area_dict, user_level_label)
    cc_timezone = area_dict['timezone']
    max_place_area = area_dict.get('max_place_area') or 1e9

    pool = mp.Pool(cpus)
    for df_access in data_access.yield_tweets_access(tweets_files_paths):
        args = (df_access, get_df_fun, valid_uids, places_geodf, langs_agg_dict,
                cells_df_list, max_place_area, cc_timezone)
        kwargs = {'min_nr_words': 4, 'cld': 'pycld2'}
        pool.apply_async(
            uagg.get_lang_loc_habits, args, kwargs, callback=collect_user_agg_res,
            error_callback=print)
    pool.close()
    pool.join()
    from_uagg_res(
        user_agg_res, cells_df_list, places_geodf, plot_langs_dict, area_dict, 
        cell_data_path_format, lang_relevant_prop=lang_relevant_prop,
        lang_relevant_count=lang_relevant_count,
        place_relevant_th=place_relevant_th, cell_relevant_th=cell_relevant_th,
        fig_dir=fig_dir)
