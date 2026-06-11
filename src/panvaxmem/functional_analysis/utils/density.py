"""Utilities for computing mrVI sample-posterior densities."""

import anndata
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scvi
from tqdm import tqdm

from panvaxmem.functional_analysis.utils.sample_metadata import Sample


def _mixture_log_prob_single(u_loc, u_cov, sample_locs, sample_covs):
    def component_log_pdf(sample_loc, sample_cov):
        return jax.scipy.stats.multivariate_normal.logpdf(
            u_loc,
            sample_loc,
            sample_cov + u_cov,
        )

    component_log_probs = jax.vmap(component_log_pdf)(sample_locs, sample_covs)
    return -jnp.log(sample_locs.shape[0]) + jax.scipy.special.logsumexp(component_log_probs)


@jax.jit
def _batch_mixture_log_prob_with_uncertainty(qu_locs, qu_covs, sample_locs, sample_covs):
    return jax.vmap(_mixture_log_prob_single, in_axes=(0, 0, None, None), out_axes=0)(
        qu_locs,
        qu_covs,
        sample_locs,
        sample_covs,
    )


def compute_u_mean_and_scales(
    adata: anndata.AnnData,
    model: scvi.external.MRVI,
    batch_size: int = 1024,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    idxs = np.arange(adata.n_obs)
    scdl = model._make_data_loader(
        adata=adata,
        indices=idxs,
        batch_size=batch_size,
        iter_ndarray=True,
    )

    qu_locs, qu_scales = [], []
    jit_inference_fn = model.module.get_jit_inference_fn(inference_kwargs={"use_mean": True})
    for array_dict in scdl:
        outputs = jit_inference_fn(model.module.rngs, array_dict)
        qu_locs.append(outputs["qu"].loc)
        qu_scales.append(outputs["qu"].scale)

    return jnp.concatenate(qu_locs, axis=0), jnp.concatenate(qu_scales, axis=0)


def compute_log_probs_under_sample_posterior(
    qu_locs: jnp.ndarray,
    qu_scales: jnp.ndarray,
    sample_idxs: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    qu_covs = jax.vmap(jnp.diag)(qu_scales**2)
    sample_locs = qu_locs[sample_idxs]
    sample_covs = qu_covs[sample_idxs]

    log_probs = []
    n_splits = max(int(np.ceil(qu_locs.shape[0] / batch_size)), 1)
    for batch_idxs in np.array_split(np.arange(qu_locs.shape[0]), n_splits):
        batch_locs = qu_locs[batch_idxs]
        log_probs.append(
            _batch_mixture_log_prob_with_uncertainty(
                batch_locs,
                qu_covs[batch_idxs],
                sample_locs,
                sample_covs,
            )
        )

    return np.array(jnp.concatenate(log_probs, axis=0))


def compute_log_probs_under_all_sample_posteriors(
    qu_locs: jnp.ndarray,
    qu_scales: jnp.ndarray,
    obs: pd.DataFrame,
    samples: list[Sample],
    donor_key: str = "donor",
    time_key: str = "time_int",
    batch_size: int = 4096,
) -> dict[Sample, np.ndarray]:
    log_probs_per_sample = {}
    for sample in tqdm(samples):
        sample_idxs = np.where((obs[donor_key] == sample.donor) & (obs[time_key] == sample.time))[0]
        assert len(sample_idxs) > 0, f"No cells found for sample {sample}"
        log_probs = compute_log_probs_under_sample_posterior(
            qu_locs,
            qu_scales,
            sample_idxs,
            batch_size=batch_size,
        )
        assert not np.isnan(log_probs).any(), f"NaN in log_probs for sample {sample}"
        log_probs_per_sample[sample] = log_probs
    return log_probs_per_sample
