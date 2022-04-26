"""MethylDrift

MethylDrift is a tool that identifies age-related tissue-specific epigenetic drift in a given dataset,
to compute rates of aging across patient samples, and to ultimately apply the pipeline to any tissue
of interest given user input. It provides a fast, comprehensive bioinformatics pipeline for the
quantification of epigenetic drift in large datasets. Takes DNA methylation array data or preproccesed
methylation data for any tissue of interest as input, determines the CpG sites with significant drift
effects, renders a text file with static/drifting sites, and estimates drift rate distributions.
"""

import typer
from typing import Optional
from pathlib import Path
from enum import Enum
from textwrap import dedent
import logging
import time

app = typer.Typer(help=dedent("""
    MethylDrift is a tool that identifies age-related tissue-specific epigenetic drift in a given dataset,
    to compute rates of aging across patient samples, and to ultimately apply the pipeline to any tissue
    of interest given user input.
    """))

class LoggingLevel(Enum):
    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
valid_ext = ['.csv', '.tsv', '.txt', '.fth', '.feather']

# @app.command()
def msce(
    pheno_file: Path = typer.Argument(..., help=dedent("""
        Path to CSV with sample phenotypes and metadata. Must have the following columns:\n
        file_type: str = raw, processed\n
        idat_type: str = EPIC, 450k\n
        sample_id: str = basenames for .idat files or sample IDs for processed beta values\n
        age: numeric, age of patient when sample was collected\n
        sample_group: normal, [other]\n
        [optional] pat_id: patient IDs. If there are duplicates, will run steps for longitudinal data.
        """)),
    dwell_mean: float = typer.Argument(..., min=0, help="Prior mean dwell time."),
    dwell_std: float = typer.Argument(..., min=0, help="Prior standard deviation of dwell time."),
    bval_file: Optional[Path] = typer.Argument(None, help=dedent(f"""
        Path to file of processed beta values.\n
        Note that filetype is inferred from extension and must be one of: {valid_ext}\n
        Should have a column "cg" of CpG IDs with other columns being sample IDs.\n
        """)),
    idat_dir: Optional[Path] = typer.Argument(None, help='Path to folder with .idat files, or folder with 450k/EPIC as subdirectories with corresponding .idat files.'),
    output_dir: Optional[Path] = typer.Argument(None, help='Defaults to methyldrift_results_[time].'),
    cgs_file: Optional[Path] = typer.Argument(None, help='Instead of running CpG selection, use a list of CpG IDs.'),
    from_minfi: bool = typer.Option(False, '--from-minfi', help='Whether to continue from MethylDrift\'s minfi results.'),
    logging_level: LoggingLevel = typer.Option('ERROR')
    ):
    """Estimate patient-specific tumor dwell times using a prior from multistage clonal expansion models."""

    raise NotImplementedError

    from methyldrift import lib, model, selection
    import pandas as pd

    if output_dir is None:
        timestamp = time.strftime('%y%m%d.%H.%M.%S', time.gmtime())
        output_dir = Path(f'methyldrift_results_{timestamp}')

    lib.init_logging(output_dir, logging_level)
    pheno = lib.validate_pheno_file(pheno_file, idat_dir)

    
    # Read preprocessed data
    logging.info('Reading in preprocessed beta values and converting to m-values.')
    if bval_file:
        lib.validate_bval_file(bval_file)
        processed_mvals = lib.read_bval_to_mval(bval_file)

    # Process idat files or retrieve results
    if from_minfi:
        logging.info('Retrieving minfi output.')
        minfi_mvals_path = output_dir.joinpath('minfi_mvals.fth')
        cg_ann_path = output_dir.joinpath('cg_ann.fth')
        minfi_mvals, cg_ann = lib.minfi_get_result(minfi_mvals_path, cg_ann_path)
    elif idat_dir:
        logging.info('Processing .idat files.')
        minfi_mvals_path, cg_ann_path = lib.minfi_run(pheno_file, idat_dir, output_dir)
        minfi_mvals, cg_ann = lib.minfi_get_result(minfi_mvals_path, cg_ann_path)

    # Combine m-values as necessary
    mvals: pd.DataFrame
    if idat_dir and bval_file:
        logging.info('Combining m-values from raw .idat and preprocessed data.')
        mvals = minfi_mvals.join(processed_mvals, how='inner').T
        logging.info(f'Number of common CpG features: {len(mvals.columns)}.')
    elif idat_dir:
        mvals = minfi_mvals.T
    else:
        mvals = processed_mvals.T
    

    # Partition controls and non-controls
    pheno.set_index('sample_id', drop=True, inplace=True) # (n, ...)
    ctrl_pheno = pheno[pheno['sample_group'] == 'normal']
    mlg_pheno = pheno[pheno['sample_group'] != 'normal']
    ctrl_m: pd.DataFrame = mvals.loc[ctrl_pheno.index] # (n, m)
    mlg_m: pd.DataFrame = mvals.loc[mlg_pheno.index] # (n, m)

    # Standardize
    # Necessary to imply beta_t is Cauchy distributed, also good for regression optimizers
    ctrl_m_stder, mlg_m_stder, ctrl_pheno_stder, mlg_m_stder, ctrl_age, mlg_age = (
        lib.standardize(ctrl_m, mlg_m, ctrl_pheno, mlg_pheno)
    )

    # Regress MethylDrift equation for point estimates
    logging.info('Running regressions on m-values for CpG feature selection and empirical priors.')
    ctrl_res = lib.md_lstsq('s_age', ctrl_age, ctrl_m)
    beta_t = lib.get_beta_t(mlg_m, mlg_age, ctrl_res, dwell_mean)


    # Select CpGs for downstream Bayesian model
    if cgs_file:
        cgs = pd.read_csv(cgs_file, index_col=0, header=None, names=['cg']).index # type: ignore
    else:
        logging.info('Selecting CpG sites for estimating dwell times.')
        cgs = selection.by_difference(mvals.columns, ctrl_res, beta_t)
        cgs = selection.by_normality(cgs, mlg_m)
        pct_selected = len(cgs)/len(ctrl_m.columns)*100
        logging.info(f'{len(cgs)}/{len(ctrl_m.columns)} ({pct_selected:.2f}%) of CpG features were selected for model fitting.')


    # Fit empirical priors
    # priors = model.fit_empirical_priors(cgs, mlg_m, beta_t, sojourn_time)
    md_model_args = model.get_msce_model_args(cgs, ctrl_m, mlg_m, ctrl_res, mlg_pheno)
    md_model = model.msce_model(*md_model_args)


@app.command()
def longitudinal(
    pheno_file: Path = typer.Argument(..., help=dedent("""
        Path to CSV with sample phenotypes and metadata. Must have the following columns:\n
        file_type: str = raw, processed\n
        idat_type: str = EPIC, 450k\n
        sample_id: str = basenames for .idat files or sample IDs for processed beta values\n
        age: numeric = age of patient when sample was collected\n
        sample_group: str = normal, [other]\n
        pat_id: str = patient IDs
        """)),
    bval_file: Path = typer.Argument(..., help=dedent(f"""
        Path to file of processed beta values.\n
        Note that filetype is inferred from extension and must be one of: {valid_ext}\n
        Should have a column "cg" of CpG IDs with other columns being sample IDs.\n
        """)),
    output_dir: Optional[Path] = typer.Argument(None, help='Defaults to methyldrift_results_[time].'),
    cgs_file: Optional[Path] = typer.Argument(None, help='Instead of running CpG selection, use a list of CpG IDs.'),
    logging_level: LoggingLevel = typer.Option('ERROR')
    ):
    """Use longitudinal data from maligant samples to estimate patient-specific dwell times in both cross-sectional and longitudinal data. WIP"""

    from methyldrift import lib, model, selection, longi
    import pandas as pd
    import arviz as az

    if output_dir is None:
        timestamp = time.strftime('%y%m%d.%H.%M.%S', time.gmtime())
        output_dir = Path(f'methyldrift_results_{timestamp}')

    lib.init_logging(output_dir, logging_level)
    pheno = lib.validate_pheno_file(pheno_file, longitudinal=True)


    logging.info('Reading beta values and converting to m-values.')
    mvals = lib.read_bval_to_mval(bval_file).T
    

    # Partition controls and non-controls
    pheno.set_index('sample_id', drop=True, inplace=True) # (n, ...)
    ctrl_pheno = pheno[pheno['sample_group'] == 'normal']
    mlg_pheno = pheno[pheno['sample_group'] != 'normal']
    ctrl_m: pd.DataFrame = mvals.loc[ctrl_pheno.index] # (n, m)
    mlg_m: pd.DataFrame = mvals.loc[mlg_pheno.index] # (n, m)


    logging.info('Running regressions on normal samples to estimate normal drift rates.')
    ctrl_res = lib.md_lstsq('age', ctrl_pheno, ctrl_m)


    logging.info('Running regressions on longitudinal samples to estimate (pre)malignant drift rates.')
    longi_pheno, longi_m = longi.get_longi_data(mlg_pheno, mlg_m)
    longi_params = longi.get_longi_params(longi_pheno, longi_m, ctrl_res)
    beta_t_res = longi.get_beta_t_res(longi_pheno, longi_m, longi_params, ctrl_res)


    # Select CpGs for downstream Bayesian model
    if cgs_file is None:
        logging.info('Selecting CpG sites for estimating dwell times.')
        cgs = selection.by_delta_m(mlg_m, mlg_pheno, ctrl_res, beta_t_res)
        pct_selected = len(cgs)/len(ctrl_m.columns)*100
        logging.info(f'{len(cgs)}/{len(ctrl_m.columns)} ({pct_selected:.2f}%) of CpG features were selected for model fitting.')
    else:
        logging.info(f'Using CpGs from {cgs_file}.')
        cgs: pd.Index = pd.read_csv(cgs_file, index_col=0, header=None, names=['cg']).index # type: ignore
        n_input = len(cgs)
        cgs = cgs[cgs.isin(mlg_m.columns)]
        pct_available = len(cgs)/n_input*100
        logging.info(f'{len(cgs)}/{n_input} ({pct_available})% of input CpGs are available in the dataset.')
    

    logging.info('Estimating patients\'s dwell times/ages of onset.')
    model_kwargs = model.get_longi_model_kwargs(cgs, ctrl_m, mlg_m, ctrl_res, mlg_pheno, beta_t_res)
    mcmc = model.nuts_estimate(model.longi_model, model_kwargs)
    idata = model.get_idata(mcmc, model.longi_model, model_kwargs)

    az.to_netcdf(idata, output_dir.joinpath('mcmc.nc'))


if __name__ == '__main__':
    app()