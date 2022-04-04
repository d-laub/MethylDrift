import pandas as pd
from .lib import LstsqResult, md_lstsq


def get_longi_data(mlg_pheno: pd.DataFrame, mlg_m: pd.DataFrame):
    """Get longitudinal phenotypes and m-values."""
    gt_1_tp = mlg_pheno.groupby('pat_id')['age'].nunique() > 1
    gt_1_tp_pat_ids = gt_1_tp[gt_1_tp].index
    longi_pheno: pd.DataFrame = mlg_pheno.loc[mlg_pheno['pat_id'].isin(gt_1_tp_pat_ids)].copy()
    longi_pheno['n_timepoints'] = longi_pheno.groupby('pat_id')['age'].transform('size')
    longi_m: pd.DataFrame = mlg_m.loc[longi_pheno.index]
    longi_pheno['origin_age'] = longi_pheno['age'] - longi_pheno.groupby('pat_id')['age'].transform('min')
    return longi_pheno, longi_m
    

def get_longi_params(longi_pheno: pd.DataFrame, longi_m: pd.DataFrame, ctrl_res: LstsqResult):
    """Regress out normal drift, then for each patient regress m ~ age and extract the parameters."""
    # (n,m) = (n,m) - (m,) - (m,) * (n,1)
    resid = (
        longi_m
        - ctrl_res.params['Intercept'].to_numpy() 
        - ctrl_res.params['age'].to_numpy()*longi_pheno['age'].to_numpy()[:, None]
    )
    dfs = []
    for pat_id, pat_pheno in longi_pheno.groupby('pat_id'):
        Y = resid.loc[pat_pheno.index] # (n, m)
        res = md_lstsq('age', pat_pheno, Y)
        res.params['pat_id'] = pat_id
        dfs.append(res.params)
    longi_params = pd.concat(dfs).reset_index()
    longi_params = longi_params.merge(longi_pheno[['pat_id', 'n_timepoints']].drop_duplicates(), on='pat_id', how='left')
    return longi_params


def get_beta_t_res(
    longi_pheno: pd.DataFrame,
    longi_m: pd.DataFrame,
    longi_params: pd.DataFrame,
    ctrl_res: LstsqResult,
    return_origin=False
    ):
    """Regress over all longitudinal patients after fixing them to origin, yielding beta_t.
    
    "Fixing to origin": let (0,0) = ( min(age), f(min(age)) ) where f() is the linear regressor estimated in longi_params.
    """
    resid = (
        longi_m
        - ctrl_res.params['Intercept'].to_numpy() 
        - ctrl_res.params['age'].to_numpy()*longi_pheno['age'].to_numpy()[:, None]
    )
    origin_m = resid.copy()
    for pat_id, params in longi_params.groupby('pat_id'):
        pat_pheno = longi_pheno[longi_pheno['pat_id'] == pat_id]
        idx = pat_pheno.index
        min_age = pat_pheno['age'].min()
        origin_m.loc[idx] -= params['Intercept'].to_numpy() + params['age'].to_numpy()*min_age
    beta_t_res = md_lstsq('0 + origin_age', longi_pheno, origin_m)
    if return_origin:
        return beta_t_res, origin_m
    else:
        return beta_t_res
    

def get_dwell(clock_cgs: pd.Index, longi_m: pd.DataFrame, longi_pheno: pd.DataFrame, ctrl_res: LstsqResult, origin_res: LstsqResult) -> pd.DataFrame:
    # let m be the number of clock cpgs
    clock_longi_m = longi_m[clock_cgs]
    clock_ctrl_params = ctrl_res.params.loc[clock_cgs]
    # (n,m)
    resid = (
        clock_longi_m.to_numpy() # (n,m)
        - clock_ctrl_params['Intercept'].to_numpy() # (m,)
        - clock_ctrl_params['age'].to_numpy() * longi_pheno['age'].to_numpy()[:, None] # (m,) * (n,1)
    )
    # (n,m) = (n,m) / (m,)
    dwell = resid / origin_res.params.loc[clock_cgs, 'origin_age'].to_numpy()
    dwell = pd.DataFrame(dwell, index=clock_longi_m.index, columns=clock_longi_m.columns)
    return dwell.dropna(axis=1, how='all')