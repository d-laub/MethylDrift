"""Setting up and running models in NumPyro."""

from typing import Callable, Dict
import pandas as pd
import scipy.stats as st
from .lib import LstsqResult
import jax
from jax.random import PRNGKey
import numpyro as npr
import numpyro.distributions as dist
import arviz as az


def get_msce_model_args(
    cgs: pd.Index,
    ctrl_m: pd.DataFrame,
    mlg_m: pd.DataFrame,
    ctrl_res: LstsqResult,
    mlg_pheno: pd.DataFrame
    ) -> Dict:

    # reparam for numpyro
    sig_shape, _, scale = st.gamma.fit(ctrl_m[cgs].std(0), floc=0) 
    sig_rate = 1/scale

    ids = mlg_m.index
    b0 = ctrl_res.params.loc[cgs, 'Intercept'].to_numpy()
    bn = ctrl_res.params.loc[cgs, 's_age'].to_numpy()
    age = mlg_pheno.loc[ids, ['s_age']].to_numpy()
    mvals = mlg_m.loc[ids, cgs].to_numpy()

    model_args = {
        'b0': b0,
        'bn': bn,
        'sig_shape': sig_shape,
        'sig_rate': sig_rate,
        'age': age,
        'ids': ids,
        'cgs': cgs,
        'mvals': mvals
    }
    
    return model_args


def msce_model(b0, bn, sig_shape, sig_rate, age, ids, cgs, mvals=None):
    raise NotImplementedError('Need to fix formulation of prior for bar_d and d.')
    plate_ids = npr.plate('ids', len(ids), dim=-2)
    plate_cgs = npr.plate('cgs', len(cgs), dim=-1)
    
    bar_d = npr.sample('bar_d', dist.Normal(0, 1))
    
    with plate_ids:
        d = npr.sample('d', dist.Exponential(1/bar_d))

    with plate_cgs:
        sigma = dist.Gamma(sig_shape, sig_rate)
    
    with plate_ids, plate_cgs:
        bt = npr.sample('beta_t', dist.Cauchy())
        # (n,m) = (m) + (m,) * (n,1) + (n,m) * (n,1)
        mu = npr.deterministic('mu', b0 + bn * age + bt * d)
        npr.sample('mvals', dist.Normal(mu, sigma), obs=mvals)


def get_longi_model_kwargs(
    cgs: pd.Index,
    ctrl_m: pd.DataFrame,
    mlg_m: pd.DataFrame,
    ctrl_res: LstsqResult,
    mlg_pheno: pd.DataFrame,
    beta_t_res: LstsqResult,
    ) -> Dict:

    # reparam for numpyro
    sig_shape, _, scale = st.gamma.fit(ctrl_m[cgs].std(0), floc=0) 
    sig_rate = 1/scale

    ids = mlg_m.index
    b0 = ctrl_res.params.loc[cgs, 'Intercept'].to_numpy()
    bn = ctrl_res.params.loc[cgs, 'age'].to_numpy()
    bt_mean = beta_t_res.params.loc[cgs, 'origin_age'].mean()
    bt_sigma = beta_t_res.params.loc[cgs, 'origin_age'].std()
    age = mlg_pheno.loc[ids, ['age']].to_numpy()
    age_dx = mlg_pheno.loc[ids, ['age_dx']].to_numpy()
    mvals = mlg_m.loc[ids, cgs].to_numpy()

    model_args = {
        'b0': b0,
        'bn': bn,
        'sig_shape': sig_shape,
        'sig_rate': sig_rate,
        'bt_mean': bt_mean,
        'bt_sigma': bt_sigma,
        'age': age,
        'age_dx': age_dx,
        'ids': ids,
        'cgs': cgs,
        'mvals': mvals
    }
    
    return model_args


def longi_model(b0, bn, sig_shape, sig_rate, bt_mean, bt_sigma, age, age_dx, ids, cgs, mvals=None):
    plate_ids = npr.plate('ids', len(ids), dim=-2)
    plate_cgs = npr.plate('cgs', len(cgs), dim=-1)
    
    with plate_ids:
        d = npr.sample('d', dist.Uniform(age-age_dx, age_dx))

    with plate_cgs:
        sigma = npr.sample('sigma', dist.Gamma(sig_shape, sig_rate))
    
    with plate_ids, plate_cgs:
        bt = npr.sample('beta_t', dist.Normal(bt_mean, bt_sigma))
        # (n,m) = (m) + (m,) * (n,1) + (n,m) * (n,1)
        mu = npr.deterministic('mu', b0 + bn * age + bt * d)
        npr.sample('mvals', dist.Normal(mu, sigma), obs=mvals)


def map_estimate(
    model: Callable,
    model_kwargs: Dict,
    step_size: float = 5e-4,
    n_steps: int = int(1e4),
    seed: int = 0
    ):
    optimizer = npr.optim.Adam(step_size=step_size)
    guide = npr.infer.autoguide.AutoDelta(model)
    svi = npr.infer.SVI(model, guide, optimizer, loss=npr.infer.Trace_ELBO())
    svi_res = svi.run(PRNGKey(seed), n_steps, **model_kwargs)
    return svi_res


def nuts_estimate(
    model: Callable,
    model_kwargs: Dict,
    num_warmup: int = 1000,
    num_samples: int = 500,
    num_chains: int = 2,
    seed: int = 0
    ):
    nuts = npr.infer.NUTS(model, init_strategy=npr.infer.init_to_median)
    mcmc = npr.infer.MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(PRNGKey(seed), **model_kwargs)
    return mcmc

def get_idata(mcmc: npr.infer.MCMC, model=None, model_kwargs=None, predictive=False, num_samples=500, seed=0):
    if predictive:
        assert model is not None and model_kwargs is not None, 'Need model and model_kwargs if adding the posterior predictive to idata.'
        post_samp = mcmc.get_samples()
        post_pred = npr.infer.Predictive(model, post_samp, num_samples=num_samples)(PRNGKey(seed), **model_kwargs)
    else:
        post_pred = None
    idata = az.from_numpyro(mcmc, posterior_predictive=post_pred)
    return idata