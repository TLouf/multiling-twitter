'''
This module contains functions aimed at creating a number of configuration
dictionaries, thus enabling a fast and automatic configuration adapted to every
area of interest, whether it's a whole country, or only a region, or whether
2, 3 or more languages are spoken by locals there. The basic configuration for
each country is simply in a JSON file in data/external, read into the
dictionary `countries_study_data` then used to make the `area_dict`.
'''
from itertools import combinations
import pycld2


LANGS_DICT = dict([(lang[1], lang[0].lower().capitalize())
                   for lang in pycld2.LANGUAGES])


def area_dict(countries_study_data, cc, region=None):
    if region:
        region_dict = countries_study_data[cc]['regions'][region].copy()
    else:
        region_dict = countries_study_data[cc].copy()
    region_dict['cc'] = cc
    region_dict['region'] = region
    return region_dict


def shapefile_dict(region_dict, cc, region=None,
                   default_shapefile='CNTR_RG_01M_2016_4326.shp'):
    shapefile_res = {
        'cc': cc,
        'region': region,
        'name': region_dict.get('shapefile_name', default_shapefile),
        'col': region_dict.get('shapefile_name_col', 'FID'),
        'val': region_dict.get('shapefile_name_val', cc)}
    return shapefile_res


def langs_dict(region_dict, level_lang_label_format):
    plot_langs_list = region_dict['local_langs']
    plot_langs_dict = {}
    for plot_lang in plot_langs_list:
        readable_lang = LANGS_DICT[plot_lang]
        lang_count_col = f'count_{plot_lang}'
        level_lang_label = level_lang_label_format.format(readable_lang)
        lang_count_label = f'Number of {level_lang_label} in the cell'
        lang_dict = {'count_col': lang_count_col,
                     'count_label': lang_count_label,
                     'grp_label': level_lang_label,
                     'readable': readable_lang}
        plot_langs_dict[plot_lang] = lang_dict
    return plot_langs_dict


def linguals_dict(region_dict):
    plot_linguals_dict = {}
    all_mulitiling_types = {1: 'mono', 2: 'bi', 3: 'tri', 4: 'quadri'}
    plot_langs_list = region_dict['local_langs']
    lings_list = []
    # Get all possible kinds of multilinguals in lings_list
    for L in range(1, len(plot_langs_list)+1):
        for subset in combinations(sorted(plot_langs_list), L):
            lings_list.append(''.join(subset))

    for ling in lings_list:
        ling_count_col = f'count_ling_{ling}'
        # We extract every 2-letter language code from ling
        nr_langs_in_ling = len(ling)//2
        langs_in_ling = [ling[2*k:2*(k+1)] for k in range(nr_langs_in_ling)]
        readable_langs = [LANGS_DICT[lang] for lang in langs_in_ling]
        multiling_type = all_mulitiling_types.get(nr_langs_in_ling, 'multi')
        readable_ling = '-'.join(readable_langs)
        ling_label = f"{readable_ling} {multiling_type}linguals"
        ling_count_label = f'Number of {ling_label} in the cell'
        ling_dict = {'count_col': ling_count_col,
                     'count_label': ling_count_label,
                     'readable': readable_ling,
                     'grp_label': ling_label}
        plot_linguals_dict['ling_' + ling] = ling_dict
    return plot_linguals_dict


def multi_mono_dict(plot_lings_dict):
    plot_multi_mono_dict = {}
    for ling, ling_dict in plot_lings_dict.items():
        if len(ling.split('_')[1]) == 2:
            mono_dict = {'count_col': ling_dict['count_col'],
                         'count_label': ling_dict['count_label'],
                         'readable': ling_dict['readable'],
                         'grp_label': ling_dict['grp_label']}
            plot_multi_mono_dict['mono_' + ling] = mono_dict

    multi_dict = {'count_col': 'multi_count',
                 'count_label': 'Number of multilinguals in the cell',
                 'readable': 'multilinguals',
                 'grp_label': 'multilinguals'}
    plot_multi_mono_dict['multi'] = multi_dict
    return plot_multi_mono_dict
