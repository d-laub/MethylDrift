import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import logging
import time
import numpy as np
import numpy.linalg as la
import scipy.stats as st
import pandas as pd
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import pandera as pa
from patsy.highlevel import dmatrix


def init_logging(out_folder: Path, level='ERROR'):
    """Only call this once per logging module usage."""
    timestamp = time.strftime('%y%m%d.%H.%M.%S', time.gmtime())
    log_path = out_folder.joinpath('logs', f'{timestamp}.log')
    if not log_path.parent.exists(): log_path.parent.mkdir()

    level = getattr(logging, level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f'Invalid log level: {level}')

    logging.basicConfig(filename=log_path, filemode='w', level=level)


def infer_delim(csv_file: Path) -> Optional[str]:
    """Infer the delimiter for a .csv file based on the file extension."""
    if csv_file.suffix == '.csv':
        delim = ','
    elif csv_file.suffix in {'.tsv', '.txt'}:
        delim = '\t'
    else:
        delim = None
    return delim


def validate_pheno_file(pheno_file: Path, idat_folder=None, longitudinal=False) -> pd.DataFrame:
    schema = pa.DataFrameSchema({
        'sample_group': pa.Column(str, [
                pa.Check(lambda s: (s == 'normal').any()), # have at least one normal sample
                pa.Check(lambda s: ~(s == 'normal').all()) # have at least one malignant sample
            ], coerce=True),
        'sample_id': pa.Column(str, coerce=True, unique=True),
        'age': pa.Column(float, coerce=True),
    })

    if idat_folder is not None:
        minfi_schema = {
            'file_type': pa.Column(str, pa.Check.isin(['raw', 'processed'])),
            'idat_type': pa.Column(str, pa.Check.isin(['EPIC', '450k'])),
        }
        schema = schema.update_columns(minfi_schema)
    
    if longitudinal:
        schema = schema.update_columns({'pat_id': pa.Column(str, coerce=True, unique=True)})

    delim = infer_delim(pheno_file)
    if delim is not None:
        pheno = pd.read_csv(pheno_file, sep=delim)
    else:
        RuntimeError(f"Unrecognized file extension for phenotype file: {pheno_file}. Must be one of .csv, .tsv, or .txt")
    schema.validate(pheno, inplace=True)

    return pheno


def validate_bval_file(bval_file: Path):
    valid_ext = ['.csv', '.tsv', '.txt', '.fth', '.feather']
    if bval_file.suffix not in valid_ext:
        raise RuntimeError(f'Invalid extension for bval_file. Got {bval_file.suffix}. Expected one of {valid_ext}.')


def validate_longitudinal(pheno: pd.DataFrame):
    if not pheno['pat_id'].duplicated().any():
        raise RuntimeError('No patients with longitudinal data found based on duplicate patient IDs.')


def minfi_run(pheno_file: Path, idat_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    """Run minfi processing script to generate m-values from .idat files.
    
    Parameters
    ----------
    info_file : Path
        Path to .csv of patient info.
    idat_dir : Path
        Path to folder containing raw .idat files.
    out_dir: Path
        Path to folder for output.

    Returns
    -------
    minfi_mvals_path : Path
        Path to m-values.
    cg_ann_path : Path
        Path to CpG annotations such as genomic coordinates and whether it is an island or singleton.
    """

    pheno = pd.read_csv(pheno_file)
    if 'Basename' not in pheno.columns:
        raise RuntimeError('''\
            The column "Basename" was not found in the patient info file.
            "Basename" should be the concatenation of each sample's Sentrix ID and Sentrix Array numbers.
            i.e. <Sentrix ID>_<Sentrix Array>
            ''')
    raw = pheno[pheno['file_type'] == 'raw']
    raw.rename(columns={'sample_id': 'Basename'}, inplace=True)
    
    # checking which idat types exist : 450k / EPIC
    epic = (raw['idat_type'] == 'EPIC').any()
    i450k = (raw['idat_type'] == '450k').any()

    # minfi processing
    if i450k and epic:
        subprocess.run(f'Rscript ../R/run_minfi.R Both {idat_dir} {raw} {out_dir}'.split(), check=True)
    elif i450k:
        subprocess.run(f'Rscript ../R/run_minfi.R 450k {idat_dir} {raw} {out_dir}'.split(), check=True)
    elif epic:
        subprocess.run(f'Rscript ../R/run_minfi.R EPIC {idat_dir} {raw} {out_dir}'.split(), check=True)
    

    cg_ann_path = out_dir.joinpath('cg_ann.fth')
    minfi_mvals_path = out_dir.joinpath('minfi_mvals.fth')

    return minfi_mvals_path, cg_ann_path


def minfi_get_result(minfi_mvals_file: Path, cg_ann_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load results from minfi."""
    cg_ann = pd.read_feather(cg_ann_file)
    minfi_mvals = pd.read_feather(minfi_mvals_file)
    minfi_mvals.set_index('cg', drop=True, inplace=True)
    return minfi_mvals, cg_ann


def read_bval_to_mval(bval_file: Path) -> pd.DataFrame:
    delim = infer_delim(bval_file)
    if delim is None:
        bvals = pd.read_feather(bval_file)
    else:
        bvals = pd.read_csv(bval_file, sep=delim)
    if 'cg' not in bvals.columns:
        raise RuntimeError('Column "cg" not found. See help for --bval-file and check if \
            the format of processed beta values conforms to input specifications.')
    bvals.set_index('cg', drop=True, inplace=True)
    mvals: pd.DataFrame = np.log2(bvals) - np.log2(1-bvals)
    mvals.dropna(axis='index', inplace=True)
    return mvals


def standardize(ctrl_m: pd.DataFrame, mlg_m: pd.DataFrame, ctrl_pheno: pd.DataFrame, mlg_pheno: pd.DataFrame):
    ctrl_m_stder = StandardScaler()
    mlg_m_stder = StandardScaler()
    ctrl_pheno_stder = StandardScaler()
    mlg_pheno_stder = StandardScaler()
    s_ctrl_m = pd.DataFrame(
        ctrl_m_stder.fit_transform(ctrl_m),
        index=ctrl_m.index,
        columns=ctrl_m.columns
    )
    s_mlg_m = pd.DataFrame(
        mlg_m_stder.fit_transform(mlg_m),
        index=mlg_m.index,
        columns=mlg_m.columns
    )
    ctrl_age = pd.DataFrame(
        ctrl_pheno_stder.fit_transform(ctrl_pheno[['age']]),
        index=ctrl_pheno.index,
        columns=['s_age']
    )
    mlg_age = pd.DataFrame(
        mlg_pheno_stder.fit_transform(mlg_pheno[['age']]),
        index=mlg_pheno.index,
        columns=['s_age']
    )

    return ctrl_m_stder, mlg_m_stder, ctrl_pheno_stder, mlg_pheno_stder, s_ctrl_m, s_mlg_m, ctrl_age, mlg_age


@dataclass
class LstsqResult:
    params: pd.DataFrame
    pvals: pd.DataFrame
    qvals: pd.DataFrame

def md_lstsq(formula: str, X: pd.DataFrame, Y: pd.DataFrame) -> LstsqResult:
    """Compute the least-squares solution to AX = Y as well as p- and q-values (FDR-BH corrected).
    
    Parameters
    ----------
    formula : str
        Formula to pass to patsy.dmatrix, applied to X.
    X : pandas.DataFrame
    Y : pandas.DataFrame

    Returns
    -------
    LstsqResult : dataclass
        Container with attributes: params, pvals, qvals.
    """

    if len(X) != len(Y):
        raise RuntimeError(f'Number of samples in X ({len(X)}) and Y ({len(Y)}) don\'t match.')

    # setup design matrix
    X = dmatrix(formula, X, return_type='dataframe')
    
    # preprocess features
    x_fts = X.columns
    y_fts = Y.columns
    n = len(X)
    dof = n-len(x_fts)
    X_np = X.to_numpy()
    Y_np = Y.to_numpy()

    # regress
    params, *_ = la.lstsq(X_np, Y_np, rcond=None)
    params = params.T # (m, f)
    
    # t-tests on coefficients
    resid_var = (Y_np - X_np @ params.T).var(0)
    se = np.sqrt(resid_var[:, None]*np.diag(la.inv(X_np.T @ X_np)))
    t_stat = params/se
    tdist = st.t(dof, 0, 1)
    pvals = 2*np.minimum(tdist.cdf(t_stat), 1-tdist.cdf(t_stat))
    _, qvals, *_ = multipletests(pvals.flatten(), method='fdr_bh')
    qvals = qvals.reshape(pvals.shape)

    # results
    params = pd.DataFrame(params, index=y_fts, columns=x_fts)
    pvals = pd.DataFrame(pvals, index=y_fts, columns=x_fts)
    qvals = pd.DataFrame(qvals, index=y_fts, columns=x_fts)
    return LstsqResult(params=params, pvals=pvals, qvals=qvals)


def get_beta_t(mlg_mvals: pd.DataFrame, mlg_pheno: pd.DataFrame, ctrl_res: LstsqResult, dwell_mean: float) -> pd.DataFrame:
    # (n,m) = (n,m) - (m,) - (m,) * (n,1)
    resid = (
        mlg_mvals
        - ctrl_res.params['Intercept'].to_numpy()
        - ctrl_res.params['age'].to_numpy() * mlg_pheno['age'].to_numpy()[:, None]
    )
    beta_t = resid / dwell_mean
    beta_t = pd.DataFrame(beta_t, index=mlg_mvals.index, columns=mlg_mvals.columns)
    return beta_t.dropna(axis=1, how='all')