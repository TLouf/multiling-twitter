import logging
import numpy as np
import pandas as pd
import geopandas as geopd

LOGGER = logging.getLogger(__name__)

def simu_wrapper(user_df, cells_in_area_df, model_fun, lings, t_steps,
                 prestige_A=None, simu_step_file_format=None, hist_cell_df=None,
                 work_add_count_A=0, work_add_count_B=0, **model_fun_kwargs):
    '''
    Simulation wrapper, iterating `t_steps` time the model of `model_fun` over
    the agents in `user_df`. Logs the global counts and saves the cell
    aggregated counts at each step, and updates `user_df` along the way.
    prestige_A must be a dict if already initialised, regardless if spatially
    dependent or not. If not supplied, it must also be that hist_cell_df is None
    and then it's initialised.
    '''
    LOGGER.info(user_df.groupby('ling').size())
    # if isinstance(work_add_count_A, pd.Series):
    #     user_df = user_df.join(work_add_count_A.rename('work_add_count_A'), on='work_cell_id')
    # if isinstance(work_add_count_B, pd.Series):
    #     user_df = user_df.join(work_add_count_B.rename('work_add_count_B'), on='work_cell_id')
    # user_df = user_df.fillna(0)
    # res and work cells don't change so we can recalculate them from an
    # iterated user_df.
    res_grouper = user_df.groupby('res_cell_id')
    cell_res = res_grouper.size()
    work_grouper = user_df.groupby('work_cell_id')
    cell_workers = (
        work_grouper.size()
                    .add(work_add_count_A + work_add_count_B, fill_value=0)
                    .rename_axis('work_cell_id'))
    cells_counts_geodf = cells_in_area_df[['geometry']].copy()
    for ling in lings:
        cells_counts_geodf['count_'+ling] = None
    if hist_cell_df is None:
        # then cell res, cell workers
        hist_cell_df = [user_to_cell(user_df, cells_counts_geodf)]

        if prestige_A is None:
            prestige_A = {}
            mono_A_mask = user_df['ling'] == lings[0]
            mono_B_mask = user_df['ling'] == lings[1]
            sigma_A, sigma_B, global_sigma_A, global_sigma_B = get_sigmas(
                user_df, mono_A_mask, mono_B_mask, cell_res)
            prestige_A['res'] = get_prestige(
                sigma_A, sigma_B, global_sigma_A, global_sigma_B)
            sigma_A, sigma_B, global_sigma_A, global_sigma_B = get_sigmas(
                user_df, mono_A_mask, mono_B_mask, cell_workers)
            prestige_A['work'] = get_prestige(
                sigma_A, sigma_B, global_sigma_A, global_sigma_B)

    t_0 = len(hist_cell_df) // 2
    for t in range(t_steps):
        print(f'* step {t+t_0} *', end='\r')
        LOGGER.info(f'* step {t+t_0} *')
        # at home
        user_df = model_step_wrapper(
            user_df, cell_res, model_fun, lings,
            prestige_A=prestige_A['res'], **model_fun_kwargs)
        cell_step_res = user_to_cell(user_df, cells_counts_geodf)
        if simu_step_file_format:
            N_step = len(hist_cell_df)
            simu_res_file_path = simu_step_file_format.format(t_step=N_step)
            cell_step_res.drop(columns=['geometry']).to_csv(simu_res_file_path)
        hist_cell_df.append(cell_step_res)
        LOGGER.info(user_df.groupby('ling').size())

        # at work
        user_df = model_step_wrapper(
            user_df, cell_workers, model_fun, lings,
            prestige_A=prestige_A['work'], add_count_A=work_add_count_A,
            add_count_B=work_add_count_B, **model_fun_kwargs)
        cell_step_res = user_to_cell(user_df, cells_counts_geodf)
        if simu_step_file_format:
            simu_res_file_path = simu_step_file_format.format(t_step=N_step+1)
            cell_step_res.drop(columns=['geometry']).to_csv(simu_res_file_path)
        hist_cell_df.append(cell_step_res)
        LOGGER.info(user_df.groupby('ling').size())

    return user_df, hist_cell_df, prestige_A


def model_step_wrapper(user_df, cell_pop, model_fun, lings,
                       prestige_A=0.5, add_count_A=0,
                       add_count_B=0, **model_fun_kwargs):
    '''
    Wrapper for a step of a model `model_fun`, updating the sigmas, the prestige
    if required, and passing all required parameters to `model_fun`.
    '''
    # ling_A corresponds to lings[0], and ling_B to lings[1]
    mono_A_mask = user_df['ling'] == lings[0]
    mono_B_mask = user_df['ling'] == lings[1]
    cell_id_col = cell_pop.index.name
    all_cells = cell_pop.index
    sigma_A, sigma_B, global_sigma_A, global_sigma_B = get_sigmas(
        user_df, mono_A_mask, mono_B_mask, cell_pop,
        add_count_A=add_count_A, add_count_B=add_count_B)
    if prestige_A is None:
        prestige_A = get_prestige(sigma_A, sigma_B, global_sigma_A,
                                  global_sigma_B)
    cell_data = pd.DataFrame(
        {'sigma_A': sigma_A, 'sigma_B': sigma_B, 'prestige_A': prestige_A},
        index=all_cells).fillna(0)
    user_df = user_df.join(cell_data, on=cell_id_col)
    user_df = model_fun(user_df, mono_A_mask, mono_B_mask, lings,
                        **model_fun_kwargs)
    user_df = user_df.drop(columns=['sigma_A', 'sigma_B', 'prestige_A'])
    return user_df


def model_step_wrapper_partial_mob(user_df, model_fun, lings, static_prop=0.5,
                                   prestige_A=0.5, **model_fun_kwargs):
    '''
    Step wrapper in which agents only have a given probability `static_prop` of
    going to their work cell.
    '''
    # ling_A corresponds to lings[0], and ling_B to lings[1]
    mono_A_mask = user_df['ling'] == lings[0]
    mono_B_mask = user_df['ling'] == lings[1]
    user_df['draw'] = np.random.random(user_df.shape[0])
    user_df['cell_id'] = user_df['res_cell_id']
    user_df.loc[user_df['draw'] < static_prop, 'cell_id'] = 'res_cell_id'
    cell_pop = user_df.groupby('cell_id').size()
    all_cells = cell_pop.index
    sigma_A, sigma_B, global_sigma_A, global_sigma_B = get_sigmas(
        user_df, mono_A_mask, mono_B_mask, cell_pop)
    if prestige_A is None:
        prestige_A = get_prestige(sigma_A, sigma_B, global_sigma_A,
                                  global_sigma_B)
    cell_data = pd.DataFrame(
        {'sigma_A': sigma_A, 'sigma_B': sigma_B, 'prestige_A': prestige_A},
        index=all_cells).fillna(0)
    user_df = user_df.join(cell_data, on='cell_id')
    user_df = model_fun(user_df, mono_A_mask, mono_B_mask, lings,
                        **model_fun_kwargs)
    user_df = user_df.drop(columns=['sigma_A', 'sigma_B', 'prestige_A'])
    return user_df


def monomodel_step(user_df, A_mask, B_mask, lings, rate=1, a=1):
    '''
    Implements the monolinguals model as described in Abrams and Strogatz, 2003.
    '''
    prestige_A = user_df['prestige_A']
    prestige_B = 1 - prestige_A
    user_df['draw'] = np.random.random(user_df.shape[0])
    # faster to make the mask on the whole index, even though it might seem
    # more sensible to filter out A and B first
    A_to_B_mask = A_mask & (
        user_df['draw'] < rate * prestige_B * (user_df['sigma_B']))**a
    B_to_A_mask = B_mask & (
        user_df['draw'] < rate * prestige_A * (user_df['sigma_A']))**a
    user_df.loc[A_to_B_mask, 'ling'] = lings[1]
    user_df.loc[B_to_A_mask, 'ling'] = lings[0]
    return user_df


def bimodel_step(user_df, A_mask, B_mask, lings, rate=1, a=1):
    '''
    Implements the bilingual model as described in Castello, 2006.
    '''
    prestige_A = user_df['prestige_A']
    prestige_B = 1 - prestige_A
    user_df['draw'] = np.random.random(user_df.shape[0])
    AB_mask = ~(A_mask | B_mask)
    # faster to make the mask on the whole index, even though it might seem
    # more sensible to filter out A and B first
    AB_to_A_mask = AB_mask & (
        user_df['draw'] < rate * prestige_A * (1-user_df['sigma_B'])**a)
    AB_to_B_mask = AB_mask & (
        user_df['draw'] > 1 - rate * prestige_B * (1-user_df['sigma_A'])**a)
    A_to_AB = user_df['draw'] < rate * prestige_B * (user_df['sigma_B'])**a
    B_to_AB = user_df['draw'] < rate * prestige_A * (user_df['sigma_A'])**a
    to_AB_mask = (A_mask & A_to_AB) | (B_mask & B_to_AB)
    user_df.loc[AB_to_A_mask, 'ling'] = lings[0]
    user_df.loc[AB_to_B_mask, 'ling'] = lings[1]
    user_df.loc[to_AB_mask, 'ling'] = lings[2]
    return user_df


def og_minett_step(user_df, A_mask, B_mask, lings, rate=1, a=1, mu=0.01):
    '''
    Implements the bilingual model as described in Minett and Wang, 2008.
    '''
    prestige_A = user_df['prestige_A']
    prestige_B = 1 - prestige_A
    user_df['death_draw'] = np.random.random(user_df.shape[0])
    dies_mask = user_df['death_draw'] < mu
    user_df['draw'] = np.random.random(user_df.shape[0])
    AB_dies_mask = ~(A_mask | B_mask) & dies_mask
    # faster to make the mask on the whole index, even though it might seem
    # more sensible to filter out A and B first
    AB_to_A_mask = AB_dies_mask & (
        user_df['draw'] < rate * prestige_A * (user_df['sigma_A'])**a)
    AB_to_B_mask = AB_dies_mask & (
        user_df['draw'] > 1 - rate * prestige_B * (user_df['sigma_B'])**a)
    A_to_AB = user_df['draw'] < rate * prestige_B * (user_df['sigma_B'])**a
    B_to_AB = user_df['draw'] < rate * prestige_A * (user_df['sigma_A'])**a
    to_AB_mask = (~dies_mask) & ((A_mask & A_to_AB) | (B_mask & B_to_AB))
    user_df.loc[AB_to_A_mask, 'ling'] = lings[0]
    user_df.loc[AB_to_B_mask, 'ling'] = lings[1]
    user_df.loc[to_AB_mask, 'ling'] = lings[2]
    return user_df


def minett_step(user_df, A_mask, B_mask, lings, rate=1, a=1, mu=0.01):
    '''
    Implements the bilingual model as described in Minett and Wang, 2008,
    with the modification that bilinguals influence the process of forgetting
    a language, as in Castello 2006.
    '''
    prestige_A = user_df['prestige_A']
    prestige_B = 1 - prestige_A
    user_df['death_draw'] = np.random.random(user_df.shape[0])
    dies_mask = user_df['death_draw'] < mu
    user_df['draw'] = np.random.random(user_df.shape[0])
    AB_dies_mask = ~(A_mask | B_mask) & dies_mask
    # faster to make the mask on the whole index, even though it might seem
    # more sensible to filter out A and B first
    AB_to_A_mask = AB_dies_mask & (
        user_df['draw'] < rate * prestige_A * (1-user_df['sigma_B'])**a)
    AB_to_B_mask = AB_dies_mask & (
        user_df['draw'] > 1 - rate * prestige_B * (1-user_df['sigma_A'])**a)
    A_to_AB = user_df['draw'] < rate * prestige_B * (user_df['sigma_B'])**a
    B_to_AB = user_df['draw'] < rate * prestige_A * (user_df['sigma_A'])**a
    to_AB_mask = (~dies_mask) & ((A_mask & A_to_AB) | (B_mask & B_to_AB))
    user_df.loc[AB_to_A_mask, 'ling'] = lings[0]
    user_df.loc[AB_to_B_mask, 'ling'] = lings[1]
    user_df.loc[to_AB_mask, 'ling'] = lings[2]
    return user_df


def bi_pref_step(user_df, A_mask, B_mask, lings, rate=1, a=1, mu=0.01, c=1,
                 q=0.5):
    '''
    Implements the bilingual preference model that we came up with.
    '''
    prestige_A = user_df['prestige_A']
    prestige_B = 1 - prestige_A
    user_df['death_draw'] = np.random.random(user_df.shape[0])
    dies_mask = user_df['death_draw'] < mu
    user_df['draw'] = np.random.random(user_df.shape[0])
    AB_dies_mask = ~(A_mask | B_mask) & dies_mask
    # faster to make the mask on the whole index, even though it might seem
    # more sensible to filter out A and B first
    sigma_AB = 1 - user_df['sigma_A'] - user_df['sigma_B']
    AB_to_A_mask = AB_dies_mask & (
        user_df['draw'] < rate * prestige_A * (
            user_df['sigma_A'] + q * sigma_AB)**a)
    AB_to_B_mask = AB_dies_mask & (
        user_df['draw'] > 1 - rate * prestige_B * (
            user_df['sigma_B'] + (1-q) * sigma_AB)**a)
    A_to_AB = user_df['draw'] < rate * c * prestige_B * (
        user_df['sigma_B'] + (1-q) * sigma_AB)**a
    A_to_AB = A_to_AB & A_mask
    B_to_AB = user_df['draw'] < rate * c * prestige_A * (
        user_df['sigma_A'] + q * sigma_AB)**a
    B_to_AB = B_to_AB & B_mask
    to_AB_mask = (~dies_mask) & (A_to_AB | B_to_AB)
    user_df.loc[AB_to_A_mask, 'ling'] = lings[0]
    user_df.loc[AB_to_B_mask, 'ling'] = lings[1]
    user_df.loc[to_AB_mask, 'ling'] = lings[2]
    return user_df


def bi_prestige_step(user_df, A_mask, B_mask, lings, rate=1, a=1):
    '''
    Implements the bilingual model as described in Castello, 2006, with the
    modification that bilinguals have their own prestige.
    '''
    prestige_A = user_df['prestige_A']
    prestige_B = user_df['prestige_B']
    prestige_AB = 1 - prestige_B - prestige_A
    user_df['draw'] = np.random.random(user_df.shape[0])
    AB_mask = ~(A_mask | B_mask)
    # faster to make the mask on the whole index, even though it might seem
    # more sensible to filter out A and B first
    AB_to_A_mask = AB_mask & (
        user_df['draw'] < rate * prestige_A * (1-user_df['sigma_B'])**a)
    AB_to_B_mask = AB_mask & (
        user_df['draw'] > 1 - rate * prestige_B * (1-user_df['sigma_A'])**a)
    A_to_AB = user_df['draw'] < rate * prestige_AB * (user_df['sigma_B'])**a
    B_to_AB = user_df['draw'] < rate * prestige_AB * (user_df['sigma_A'])**a
    to_AB_mask = (A_mask & A_to_AB) | (B_mask & B_to_AB)
    user_df.loc[AB_to_A_mask, 'ling'] = lings[0]
    user_df.loc[AB_to_B_mask, 'ling'] = lings[1]
    user_df.loc[to_AB_mask, 'ling'] = lings[2]
    return user_df


def bi_prestige_step_wrapper(user_df, cell_pop, lings, prestige_A=0.5,
                             prestige_B=0.5, **model_fun_kwargs):
    '''
    Wrapper for `bi_prestige_step`.
    '''
    # ling_A corresponds to lings[0], and ling_B to lings[1]
    mono_A_mask = user_df['ling'] == lings[0]
    mono_B_mask = user_df['ling'] == lings[1]
    cell_id_col = cell_pop.index.name
    all_cells = cell_pop.index
    sigma_A, sigma_B, _, _ = get_sigmas(
        user_df, mono_A_mask, mono_B_mask, cell_pop)
    cell_data = pd.DataFrame(
        {'sigma_A': sigma_A, 'sigma_B': sigma_B,
         'prestige_A': prestige_A, 'prestige_B': prestige_B},
        index=all_cells).fillna(0)
    user_df = user_df.join(cell_data, on=cell_id_col)
    user_df = bi_prestige_step(user_df, mono_A_mask, mono_B_mask, lings,
                               **model_fun_kwargs)
    cols = ['sigma_A', 'sigma_B', 'prestige_A', 'prestige_B']
    user_df = user_df.drop(columns=cols)
    return user_df


def bi_prestige_simu_wrapper(
        user_df, cells_in_area_df, lings, t_steps, prestige_A=None,
        prestige_B=None, simu_step_file_format=None, hist_cell_df=None,
        **model_fun_kwargs):
    '''
    Simulation wrapper for steps with bilingual prestige (see
    `bi_prestige_step`).
    '''
    LOGGER.info(user_df.groupby('ling').size())
    # res and work cells don't change so we can recalculate them from an
    # iterated user_df, np
    cell_res = user_df.groupby('res_cell_id').size()
    cell_workers = user_df.groupby('work_cell_id').size()
    cells_counts_geodf = cells_in_area_df[['geometry']].copy()
    cells_counts_geodf[['count_'+ling for ling in lings]] = None
    if hist_cell_df is None:
        # then cell res, cell workers
        hist_cell_df = [user_to_cell(user_df, cells_counts_geodf)]

        if prestige_A is None or prestige_B is None:
            prestige_A = {}
            prestige_B = {}
            mono_A_mask = user_df['ling'] == lings[0]
            mono_B_mask = user_df['ling'] == lings[1]
            prestige_A['res'], prestige_B['res'], _, _ = get_sigmas(
                user_df, mono_A_mask, mono_B_mask, cell_res)
            prestige_A['work'], prestige_B['work'], _, _ = get_sigmas(
                user_df, mono_A_mask, mono_B_mask, cell_workers)

    t_0 = len(hist_cell_df) // 2
    for t in range(t_steps):
        LOGGER.info(f'* step {t+t_0} *')
        # at home
        user_df = bi_prestige_step_wrapper(
            user_df, cell_res, lings, prestige_A=prestige_A['res'],
            prestige_B=prestige_B['res'], **model_fun_kwargs)
        cell_step_res = user_to_cell(user_df, cells_counts_geodf)
        if simu_step_file_format:
            N_step = len(hist_cell_df)
            simu_res_file_path = simu_step_file_format.format(t_step=N_step)
            cell_step_res.drop(columns=['geometry']).to_csv(simu_res_file_path)
        hist_cell_df.append(cell_step_res)
        LOGGER.info(user_df.groupby('ling').size())

        # at work
        user_df = bi_prestige_step_wrapper(
            user_df, cell_workers, lings, prestige_A=prestige_A['work'],
            prestige_B=prestige_B['work'], **model_fun_kwargs)
        cell_step_res = user_to_cell(user_df, cells_counts_geodf)
        if simu_step_file_format:
            simu_res_file_path = simu_step_file_format.format(t_step=N_step + 1)
            cell_step_res.drop(columns=['geometry']).to_csv(simu_res_file_path)
        hist_cell_df.append(cell_step_res)
        LOGGER.info(user_df.groupby('ling').size())

    return user_df, hist_cell_df, prestige_A, prestige_B


def get_prestige(sigma_A, sigma_B, global_sigma_A, global_sigma_B,
                 kind='simple', k=0.5):
    '''
    Calculates a prestige in each cell based on the given proportions `sigma_A`
    and `sigma_B` of monolinguals.
    '''
    cell_prop_speak_A = 1 - sigma_B
    cell_prop_speak_B = 1 - sigma_A
    # Simple version:
    if kind == 'simple':
        cell_prestige_A = 0.5 * (1 + cell_prop_speak_A - cell_prop_speak_B)
    # Logistic version:
    if kind == 'logistic':
        tot_prop_speak_A = 1 - global_sigma_B
        tot_prop_speak_B = 1 - global_sigma_A
        # default to 0, which will be kept if cell_prop_speak_A == 0
        cell_prestige_A = pd.Series(np.zeros(sigma_A.shape),
                                    index=sigma_A.index)
        is_A_zero = cell_prop_speak_A == 0
        is_B_zero = cell_prop_speak_B == 0
        cell_prestige_A.loc[is_B_zero] = 1
        one_zero = is_A_zero | is_B_zero
        # is_A_zero & is_B_zero shouldn't happen
        cell_prestige_A.loc[~one_zero] = 1 / (
            1 + (cell_prop_speak_A.loc[~one_zero] / tot_prop_speak_A
                * tot_prop_speak_B / cell_prop_speak_B.loc[~one_zero])**(-k))
    cell_prestige_A = cell_prestige_A.rename('prestige_A')
    return cell_prestige_A


def get_sigmas(user_df, mono_A_mask, mono_B_mask, cell_pop, add_count_A=0,
               add_count_B=0):
    '''
    Calculates the proportions of lingual groups A and B in each cell, as well
    as the global proportions, using the dataframe giving the ling group, cell
    of residence and of work of each user `user_df`. Calculates the proportion
    on either the work or residence cell, for which the total population is
    given in `cell_pop`. add_count kwargs to add external population at work step.
    '''
    cell_id_col = cell_pop.index.name
    count_A = (user_df.loc[mono_A_mask]
                      .groupby(cell_id_col)
                      .size()
                      .add(add_count_A, fill_value=0))
    count_B = (user_df.loc[mono_B_mask]
                      .groupby(cell_id_col)
                      .size()
                      .add(add_count_B, fill_value=0))
    sigma_A = (count_A / cell_pop).fillna(0)
    sigma_B = (count_B / cell_pop).fillna(0)
    global_sigma_A = count_A.sum() / cell_pop.sum()
    global_sigma_B = count_B.sum() / cell_pop.sum()
    return sigma_A, sigma_B, global_sigma_A, global_sigma_B


def opt_get_sigmas(user_df, cell_pop, 
                   add_count_A=0, add_count_B=0):
    cell_id_col = cell_pop.index.name
    
    count_A = (user_df.groupby([cell_id_col, 'ling'])['ling']
                        .transform('size'))
    
    user_df['sigma_A'] = (count_A + add_count_A) / cell_pop
    # if isinstance(add_count_B, pd.Series):
    count_B = (user_df.groupby([cell_id_col, 'ling'])['ling']
                        .transform('size'))
    user_df['sigma_B'] = (count_B + add_count_B) / cell_pop
    return user_df
    

def user_to_cell(user_df, cells_counts_geodf):
    '''
    From `user_df`, fills `cells_counts_geodf` with the counts per ling group
    and per cell of residence and returns it as a new dataframe.
    '''
    # astype(str) so that the columns types are not necessarily categories
    new_cell_df = pd.crosstab(user_df['res_cell_id'],
                              user_df['ling'].astype(str))
    ling_cols = new_cell_df.columns
    new_cell_df = (new_cell_df.assign(local_count=lambda df:
                                      sum([df[col] for col in ling_cols]))
                              .assign(total_count=lambda df: df['local_count'])
                              .rename(columns={ling_col: 'count_'+ling_col
                                               for ling_col in ling_cols}))
    counts_df = (cells_counts_geodf.drop(columns=['geometry'])
                                   .combine_first(new_cell_df)
                                   .fillna(0))
    new_cell_df = geopd.GeoDataFrame(
        counts_df, geometry=cells_counts_geodf.geometry,
        crs=cells_counts_geodf.crs)
    new_cell_df.index.name = 'res_cell_id'
    return new_cell_df
