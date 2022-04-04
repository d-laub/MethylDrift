from dataclasses import dataclass
from methyldrift import lib
import numpy as np
import pandas as pd

@dataclass
class TestData:
    ctrl: pd.DataFrame
    ctrl_pheno: pd.DataFrame
    mlg: pd.DataFrame
    mlg_pheno: pd.DataFrame

# based on population sojourn times
def gen_data_sojourn(beta_0, beta_n, beta_t, s_mu, eps_sig=1, n_samples=100, n_fts=1000, seed=0) -> TestData:
    rng = np.random.default_rng(seed)

    beta_0 = rng.normal(beta_0, 1, (1, n_fts))
    beta_n = rng.normal(beta_n, 1, (1, n_fts))
    ctrl_ages = rng.normal(45, 5, (n_samples, 1))

    beta_t = rng.normal(beta_t, 1, (1, n_fts))
    mlg_onsets = rng.normal(45, 5, (n_samples, 1))
    mlg_sojourn = rng.exponential(s_mu, (n_samples, 1))
    eps = rng.normal(0, eps_sig, (n_samples, n_fts))

    mlg_ages = mlg_onsets + mlg_sojourn

    ctrl = beta_0 + beta_n*ctrl_ages + eps
    mlg = beta_0 + beta_n*mlg_ages + beta_t*mlg_sojourn + eps

    ctrl_idx = [f'ctrl{i}' for i in range(n_samples)]
    mlg_idx = [f'mlg{i}' for i in range(n_samples)]
    cols = [f'cg{i}' for i in range(n_fts)]
    pheno_cols = ['sample_id', 'age']
    data = TestData(
        ctrl=pd.DataFrame(ctrl, index=ctrl_idx, columns=cols),
        ctrl_pheno=pd.DataFrame(dict(zip(pheno_cols, [ctrl_idx, ctrl_ages.squeeze()]))),
        mlg=pd.DataFrame(mlg, index=mlg_idx, columns=cols),
        mlg_pheno=pd.DataFrame(dict(zip(pheno_cols, [mlg_idx, mlg_ages.squeeze()])))
    )
    return data


def gen_timeseries_data_sojourn(beta_0, beta_n, beta_t, s_mu, eps_sig=1, n_samples=100, n_tps=3, n_fts=1000, seed=0) -> TestData:
    rng = np.random.default_rng(seed)

    beta_0 = rng.normal(beta_0, 1, (1, 1, n_fts))
    beta_n = rng.normal(beta_n, 1, (1, 1, n_fts))
    ctrl_ages = rng.normal(45, 5, (n_samples, 1, ))
    ctrl_eps = rng.normal(0, eps_sig, (n_samples, n_fts))

    beta_t = rng.normal(beta_t, 1, (1, 1, n_fts))
    mlg_onsets = rng.normal(45, 5, (n_samples, 1, 1))
    mlg_sojourn = rng.exponential(s_mu, (n_samples, 1, 1))
    mlg_sojourn = np.repeat(mlg_sojourn, n_tps, 1) + np.arange(n_tps)[:, None]
    mlg_eps = rng.normal(0, eps_sig, (n_samples, n_tps, n_fts))

    mlg_ages = mlg_onsets + mlg_sojourn

    ctrl = beta_0.squeeze(0) + beta_n.squeeze(0)*ctrl_ages + ctrl_eps
    # (1, 1, f) + (1, 1, f)(n, t, 1) + (1, 1, f)(n, t, 1) + (n, t, f)
    mlg = beta_0 + beta_n*mlg_ages + beta_t*mlg_sojourn + mlg_eps

    ctrl_idx = [f'ctrl{i}' for i in range(n_samples)]
    mlg_idx = [f'mlg{i}' for i in range(n_samples)]
    cols = [f'cg{i}' for i in range(n_fts)]
    pheno_cols = ['sample_id', 'age']
    data = TestData(
        ctrl=pd.DataFrame(ctrl, index=ctrl_idx, columns=cols),
        ctrl_pheno=pd.DataFrame(dict(zip(pheno_cols, [ctrl_idx, ctrl_ages.squeeze()]))),
        mlg=pd.DataFrame(mlg, index=mlg_idx, columns=cols),
        mlg_pheno=pd.DataFrame(dict(zip(pheno_cols, [mlg_idx, mlg_ages.squeeze()])))
    )
    return data