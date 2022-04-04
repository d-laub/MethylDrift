import scipy.stats as st
import seaborn as sns
import numpy as np

def plot_dist(dist: st.rv_continuous | st.rv_discrete, left=1e-3, right=None, space='linear', **kwargs):
    if right is None:
        right = 1 - left
    if space == 'linear':
        x = np.linspace(dist.ppf(left), dist.ppf(right), 200)
    elif space == 'geom':
        x = np.geomspace(dist.ppf(left), dist.ppf(right), 200)
    return sns.lineplot(x=x, y=dist.pdf(x), **kwargs)

def plot_emp_dist(data, stat='density', kde=True, bins=100, bw_adjust=0.2, **kwargs):
    return sns.histplot(data, stat=stat, kde=kde, bins=bins, kde_kws=dict(bw_adjust=bw_adjust), **kwargs)