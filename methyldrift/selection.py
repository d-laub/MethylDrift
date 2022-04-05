"""Selection of CpG features."""

from typing import Optional
import pandas as pd
import numpy as np
from .lib import LstsqResult
import scipy.stats as st
from statsmodels.stats.multitest import multipletests


def by_difference(
    cgs,
    ctrl_res: LstsqResult,
    beta_t: pd.DataFrame | LstsqResult,
    ) -> pd.Index:
    '''Select CpGs based on the differences between slopes `beta_n` and `beta_t`.'''

    if isinstance(beta_t, pd.DataFrame):
        raise NotImplementedError('Bootstrap to get p-value/q-value of difference between slope and scalar.')
        # (n,m) = (m,) - (n,m)
        abs_diff = abs(ctrl_res.params.loc[cgs, 'age'].to_numpy() - beta_t[cgs].to_numpy())
        sig_diff = np.quantile(abs_diff, 0.05, 0) > 0
    elif isinstance(beta_t, LstsqResult):
        raise NotImplementedError('P-value/q-value for comparing slopes of different regressions.')
        abs_diff = abs(ctrl_res.params['age'].to_numpy() - beta_t.params['origin_age'].to_numpy())
    
    raise NotImplementedError()
    cgs = cgs[sig_diff]
    
    return cgs


def by_quantile(
    ctrl_mvals: pd.DataFrame,
    ctrl_res: LstsqResult,
    intercept_quantile: float = 0,
    beta_n_quantile: float = 0,
    beta_t_quantile: float = 1,
    beta_t_df: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
    '''Select CpGs based on their quantiles.'''

    beta_0_lt = ctrl_res.params['Intercept'] < ctrl_res.params['Intercept'].quantile(intercept_quantile)
    beta_n_lt = ctrl_res.params['age'] < ctrl_res.params['age'].quantile(beta_n_quantile)
    cgs = ctrl_mvals.columns[beta_0_lt & beta_n_lt]

    if beta_t_df is not None:
        beta_t_means = beta_t_df.mean(0)
        beta_t_gt = beta_t_means > beta_t_means.quantile(beta_t_quantile)
        cgs = cgs[cgs.isin(beta_t_df.columns[beta_t_gt])]
    
    return cgs


def by_value(
    ctrl_mvals: pd.DataFrame,
    ctrl_res: LstsqResult,
    intercept: float = -np.inf,
    beta_n: float = -np.inf,
    beta_t: float = np.inf,
    beta_t_df: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
    '''Select CpGs based on their values.'''

    beta_0_lt = ctrl_res.params['Intercept'] < intercept
    beta_n_lt = ctrl_res.params['age'] < beta_n
    cgs = ctrl_mvals.columns[beta_0_lt & beta_n_lt]

    if beta_t_df is not None:
        beta_t_means = beta_t_df.mean(0)
        beta_t_gt = beta_t_means > beta_t
        cgs = cgs[cgs.isin(beta_t_df.columns[beta_t_gt])]
    
    return cgs


def by_normality(
    cgs: pd.Index,
    mvals: pd.DataFrame,
    ) -> pd.Index:
    '''Select CpGs based on whether they have a normal distribution in the given dataset.'''

    norm_test_res = st.normaltest(mvals, axis=0)
    _, q_val, *_= multipletests(norm_test_res.pvalue)
    mlg_m_normal = q_val > 0.05
    cgs = cgs[cgs.isin(mvals.columns[mlg_m_normal])]
    
    return cgs


def by_delta_m(
    mlg_m: pd.DataFrame,
    mlg_pheno: pd.DataFrame,
    ctrl_res: LstsqResult,
    beta_t_res: LstsqResult,
    ):
    '''Select CpGs based on whether sgn(Δm) = sgn(β_t).'''
    resid = mlg_m.to_numpy() - ctrl_res.params['Intercept'].to_numpy() - ctrl_res.params['age'].to_numpy()*mlg_pheno['age'].to_numpy()[:, None]
    resid = pd.DataFrame(resid, index=mlg_m.index, columns=mlg_m.columns)
    match_sign = np.sign(resid) == np.sign(beta_t_res.params['origin_age'])
    all_sign_match = match_sign.all(0)
    bt_sig = beta_t_res.qvals['origin_age'] < 0.05
    cgs = all_sign_match & bt_sig
    cgs = beta_t_res.params.index[cgs]
    return cgs