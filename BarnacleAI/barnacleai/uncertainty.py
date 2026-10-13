"""Design-based 95% intervals from randomly selected, manually counted tiles.

These are intervals for a *fixed photographed region's total* induced by sampling
units without replacement. They are not model-prediction confidence scores and
only make sense if the sampled tiles were chosen at random and human counts
are accurate. The t approximation may be rough with small, skewed samples.
"""
import numpy as np
from scipy.stats import t as student_t


def partition_image(h: int, w: int, rows: int = 8, cols: int = 8):
    """Cover every pixel with nearly equal-area nonoverlapping tiles."""
    if min(h, w, rows, cols) <= 0 or rows > h or cols > w:
        raise ValueError('Invalid dimensions or grid')
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    return [(int(xs[j]),int(ys[i]),int(xs[j+1]),int(ys[i+1]))
            for i in range(rows) for j in range(cols)]


def random_tile_indices(N: int, n: int = 12, seed: int = 20261012):
    if not (2 <= n <= N):
        raise ValueError('Sample size must be between 2 and number of tiles')
    return np.random.default_rng(seed).choice(N, size=n, replace=False).tolist()


def counts_per_box(centers, boxes):
    if len(centers) == 0:
        return np.zeros(len(boxes), dtype=np.int64)
    pts=np.asarray(centers)
    return np.array([int(np.count_nonzero((pts[:,0]>=x1)&(pts[:,0]<x2)&
                      (pts[:,1]>=y1)&(pts[:,1]<y2))) for x1,y1,x2,y2 in boxes])


def finite_population_interval(manual_counts, sampled_indices, N,
                               model_counts=None, confidence=.95):
    """Simple random sample difference estimator and t-approx confidence interval.

    With model_counts of length N, estimator = sum(model) + N*mean(y-model)_sample.
    Without it, estimator = N*mean(y)_sample. Both are design-unbiased for SRS.
    Variance estimate = N^2*(1-n/N)*sample_variance(residuals)/n.
    """
    idx=np.asarray(sampled_indices,dtype=int)
    y=np.asarray(manual_counts,dtype=float)
    if len(idx)!=len(y) or len(idx)<2 or len(np.unique(idx))!=len(idx):
        raise ValueError('Need at least two unique sampled tiles with human counts')
    if np.any(y<0) or not np.all(np.isfinite(y)) or min(idx)<0 or max(idx)>=N:
        raise ValueError('Counts and indices must be valid')
    if not (0<confidence<1):
        raise ValueError('Confidence must be in (0,1)')
    if model_counts is None:
        aux=np.zeros(N,dtype=float);name='random-sampling expansion'
    else:
        aux=np.asarray(model_counts,dtype=float)
        if len(aux)!=N or not np.all(np.isfinite(aux)):
            raise ValueError('Model counts must cover the full partition')
        name='model-assisted difference'
    resid=y-aux[idx]
    est=float(aux.sum()+N*resid.mean())
    n=len(idx)
    sem=float(N*np.sqrt((1-n/N)*np.var(resid,ddof=1)/n)) if n<N else 0.
    tcrit=float(student_t.ppf((1+confidence)/2,n-1)) if n<N else 0.
    half=tcrit*sem
    return dict(method=name,sampled_tiles=n,total_tiles=N,estimate=round(est,2),
                standard_error=round(sem,2),confidence=confidence,
                lower=round(max(0.,est-half),2),upper=round(max(0.,est+half),2),
                caveat='Approximate design-based interval assuming randomly sampled tiles and accurate human counts; not a neural-network confidence score.')
