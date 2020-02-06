'''
This module contains functions aimed at creating a number of configuration
dictionaries, thus enabling a fast and automatic configuration adapted to every
area of interest, whether it's a whole country, or only a region, or whether
2, 3 or more languages are spoken by locals there. The basic configuration for
each country is simply in a JSON file in data/external, read into the
dictionary `countries_study_data` then used to make the `area_dict`.
'''
import os
import pycld2
from itertools import combinations

LANGS_DICT = dict([(lang[1],lang[0].lower().capitalize())
                   for lang in pycld2.LANGUAGES])


def area_dict(countries_study_data, cc, region=None):
    if region:
        area_dict = countries_study_data[cc]['regions'][region]
    else:
        area_dict = countries_study_data[cc]
    return area_dict


def shapefile_dict(area_dict, cc, region=None,
                   default_shapefile='CNTR_RG_01M_2016_4326.shp'):
    shapefile_name = area_dict.get('shapefile_name')
    if shapefile_name is None:
        shapefile_name = default_shapefile
    if region:
        shapefile_name_col = area_dict['shapefile_name_col']
        shapefile_name_val = region
    else:
        shapefile_name_col = 'FID'
        shapefile_name_val = cc
    shapefile_dict = {
        'name': shapefile_name,
        'col': shapefile_name_col,
        'val': shapefile_name_val}
    return shapefile_dict


def langs_dict(area_dict, level_lang_label_format):
    plot_langs_list = area_dict['local_langs']
    plot_langs_dict = {}
    for plot_lang in plot_langs_list:
        readable_lang = LANGS_DICT[plot_lang]
        lang_count_col = f'count_{plot_lang}'
        lang_prop_col = f'prop_{plot_lang}'
        level_lang_label = level_lang_label_format.format(readable_lang)
        lang_count_label = f'Number of {level_lang_label} in the cell'
        lang_dict = {'count_col': lang_count_col,
                     'count_label': lang_count_label,
                     'grp_label': level_lang_label,
                     'readable': readable_lang}
        plot_langs_dict[plot_lang] = lang_dict
    return plot_langs_dict


def linguals_dict(area_dict):
    plot_linguals_dict = {}
    all_mulitling_types = {1: 'mono', 2: 'bi', 3: 'tri'}
    plot_langs_list = area_dict['local_langs']
    plot_langs_list.sort()
    lings_list = []
    nr_langs = len(plot_langs_list)
    # Get all possible kinds of multilinguals in lings_list
    for L in range(1, len(plot_langs_list)+1):
        for subset in combinations(plot_langs_list, L):
            lings_list.append(''.join(subset))

    for ling in lings_list:
        ling_count_col = f'count_ling_{ling}'
        ling_prop_col = f'prop_ling_{ling}'
        # We extract every 2-letter language code from ling
        nr_langs_in_ling = len(ling)//2
        langs_in_ling = [ling[2*k:2*(k+1)] for k in range(nr_langs_in_ling)]
        readable_langs = [LANGS_DICT[lang] for lang in langs_in_ling]
        multiling_type = all_mulitling_types[nr_langs_in_ling]
        readable_ling = '-'.join(readable_langs)
        ling_label = f"{readable_ling} {multiling_type}linguals"
        ling_count_label = f'Number of {ling_label} in the cell'
        ling_dict = {'count_col': ling_count_col,
                     'count_label': ling_count_label,
                     'readable': readable_ling,
                     'grp_label': ling_label}
        plot_linguals_dict['ling_' + ling] = ling_dict
    return plot_linguals_dict


def whole(cc, region, cell_size, level='user'):
    if level=='user':
        level_label_format = '{}-speaking users'
    else:
        level_label_format = 'tweets in {}'

    if region:
        country_name = region
    else:
        country_name = shape_df['NAME_ENGL'].iloc[0]

    area_dict = make_config.area_dict(countries_study_data, cc, region=region)
    plot_langs_dict = make_config.plot_dict(area_dict, level_label_format)
    shapefile_dict = make_config.shapefile_dict(area_dict, cc, region=region)


    shapefile_path = os.path.join(
        external_data_dir, shapefile_dict['name'], shapefile_dict['name'])
    shape_df = geopd.read_file(shapefile_path)
    shape_df = geo.extract_shape(shape_df, shapefile_dict['col'],
                                 shapefile_dict['val'])

    cell_data_path = cell_data_path_format.format('users', cc, cell_size)
    cell_plot_df = geopd.read_file(cell_data_path)
    cell_plot_df.index = cell_plot_df['cell_id']
    return cell_plot_df, shape_df, shapefile_dict, plot_langs_dict, area_dict
