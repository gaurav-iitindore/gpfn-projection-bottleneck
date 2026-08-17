#!/usr/bin/env python3
"""
proj_ext.py

Option A projection extension for the GPFN pipeline.

Design contract (the three properties a reviewer checks first):
  1. Leakage-free. Every projection is fitted on the TRAINING partition only.
     No function here ever receives a test phenotype. y is training y only.
  2. V0 is byte-identical to the released pipeline. method='pca' is delegated
     straight to the stock util.feature_selection.feature_reduce, so the V0
     GPFN number reproduces the frozen per-seed results exactly.
  3. Only the projection changes. Every method below consumes the SAME centred
     markers the stock forward() produces (center_markers -> [-0.5, 0.5]),
     returns exactly feature_length columns, and both GPFN and the PCR control
     read the SAME returned array. So any accuracy change is the projection,
     nothing else.

This mirrors the idiom already in icar_proj_sweep.py: build features with a
chosen method, then feed them to the transformer with forward()'s own block.
The only addition is a handful of new methods and a bagging helper.

New methods (all keep the frozen 100 slots):
    pls         partial least squares, 100 components. Maximises covariance
                with the phenotype instead of marker variance.
    pls_mm      pls, then rescale each component to the PCA variance profile
                the prior was fitted on (moment matching). Run both; the gap
                between pls and pls_mm is itself a result.
    screen_pca  association scan on training only, keep top_k markers, then
                PCA to 100 on that subset. Stays in-distribution for the
                transformer while letting a large-effect locus dominate a
                component instead of being smeared across the full genome.
    hybrid      n_pcs principal components plus (100 - n_pcs) screened markers
                fed directly, moment matched. Highest distribution-shift risk,
                injects marker-level signal explicitly.

Bagging is handled by bagged_feature_sets() because it returns a LIST of
feature sets; the runner evaluates GPFN once per bag and averages predictions.

Everything else (pca, correlation, g, mi, regression, ...) is delegated to the
stock feature_reduce untouched, so no released behaviour is altered.

Author: Gaurav Tiwari
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


def _stock_feature_reduce(x, y, num_features, method, eval_x=None):
    """Delegate to the released util.feature_selection.feature_reduce.
    Imported lazily so this module can be used (and sanity-checked) without the
    full SeqBreed import chain when only the new ext methods are needed."""
    from util.feature_selection import feature_reduce as _fr
    return _fr(x, y, num_features, method, eval_x=eval_x)


# ---------------------------------------------------------------------------
# helpers (training-only statistics throughout)
# ---------------------------------------------------------------------------

def _association_r2(x_train, y_train):
    """Per-marker squared Pearson correlation with the phenotype, training
    partition only. Scale-invariant, so it is unaffected by center_markers.
    Monomorphic markers get 0 (infinite denominator)."""
    xc = x_train - x_train.mean(axis=0)
    yc = y_train - y_train.mean()
    num = xc.T @ yc
    den = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    den[den == 0] = np.inf
    return (num / den) ** 2


def _match_moments(ref_train, new_train, new_test):
    """Rescale new components so their per-component standard deviation matches
    the PCA reference the prior was fitted on. Factors computed on TRAINING
    only, applied to both train and test.

    The new-component sd is floored relative to its own median so a near-constant
    component cannot blow up (dividing by a tiny sd would inflate that column and
    push the input far off the distribution the transformer was pretrained on)."""
    ref_sd = ref_train.std(axis=0)
    new_sd = new_train.std(axis=0)
    floor = max(np.median(new_sd) * 1e-3, 1e-8)
    new_sd = np.maximum(new_sd, floor)
    factor = ref_sd / new_sd
    return new_train * factor, new_test * factor


def _pca_ref(x_train, num_features, seed):
    """Reference PCA scores on the training markers, for moment matching."""
    p = PCA(n_components=num_features, random_state=seed)
    return p.fit_transform(x_train)


# ---------------------------------------------------------------------------
# extended reducer
# ---------------------------------------------------------------------------

# methods implemented in this module; everything else is delegated to stock
_EXT_METHODS = {"pls", "pls_mm", "screen_pca", "hybrid"}


def feature_reduce_ext(x, y, num_features, method, eval_x=None,
                       seed=0, top_k=2000, n_pcs=50):
    """Drop-in extension of feature_reduce.

    x, eval_x are the CENTRED markers produced by center_markers (the caller
    does the centring, exactly as stock forward() does). y is TRAINING y only.

    Returns the same shape contract as feature_reduce:
        eval_x is None  -> Ztr
        eval_x given     -> (Ztr, Zte)
    Every returned matrix has exactly num_features columns.

    Bagging is NOT handled here (it returns a list); use bagged_feature_sets().
    """
    if method not in _EXT_METHODS:
        # pca and all existing strategies: use the released code unchanged
        return _stock_feature_reduce(x, y, num_features, method, eval_x=eval_x)

    if eval_x is None:
        raise ValueError("proj_ext methods require eval_x (train/test together).")

    if method in ("pls", "pls_mm"):
        pls = PLSRegression(n_components=num_features, scale=False)
        pls.fit(x, y)
        ztr = np.asarray(pls.transform(x))
        zte = np.asarray(pls.transform(eval_x))
        if method == "pls_mm":
            ref = _pca_ref(x, num_features, seed)
            ztr, zte = _match_moments(ref, ztr, zte)
        return ztr, zte

    if method == "screen_pca":
        if top_k < num_features:
            raise ValueError(
                f"screen_pca needs top_k >= num_features ({num_features}); got {top_k}."
            )
        r2 = _association_r2(x, y)
        keep = np.argsort(r2)[-top_k:]
        p = PCA(n_components=num_features, random_state=seed)
        ztr = p.fit_transform(x[:, keep])
        zte = p.transform(eval_x[:, keep])
        # match the enriched-subset PCA scores to the full-genome PCA variance
        # profile the transformer was pretrained on, so the input stays
        # in-distribution (without this GPFN collapses even though PCR does not)
        ref = _pca_ref(x, num_features, seed)
        ztr, zte = _match_moments(ref, ztr, zte)
        return ztr, zte

    if method == "hybrid":
        n_mark = num_features - n_pcs
        if n_mark <= 0:
            raise ValueError(f"hybrid needs n_pcs < num_features ({num_features}).")
        p = PCA(n_components=n_pcs, random_state=seed)
        ptr = p.fit_transform(x)
        pte = p.transform(eval_x)
        r2 = _association_r2(x, y)
        keep = np.argsort(r2)[-n_mark:]
        mtr, mte = x[:, keep].astype(np.float64).copy(), eval_x[:, keep].astype(np.float64).copy()
        # z-score the screened markers on TRAIN with a floored sd. Screened
        # top-effect markers are often low-MAF (tiny sd), so per-column sd
        # matching would inflate them into extreme values that wreck both PCR
        # and GPFN. Standardise first, then put the block on the PC scale with a
        # SINGLE shared scalar, and clip so a rare out-of-range dosage on test
        # cannot dominate.
        mu = mtr.mean(axis=0)
        sd = mtr.std(axis=0)
        sd = np.maximum(sd, max(np.median(sd), 1e-8))
        mtr = (mtr - mu) / sd
        mte = (mte - mu) / sd
        scale = float(np.median(ptr.std(axis=0)))
        clip = 8.0
        mtr = np.clip(mtr, -clip, clip) * scale
        mte = np.clip(mte, -clip, clip) * scale
        return np.hstack([ptr, mtr]), np.hstack([pte, mte])

    raise UserWarning(f"Unhandled ext method {method}")


def bagged_feature_sets(x, eval_x, num_features, n_bags=10, seed=0):
    """Feature bagging. Split markers into n_bags disjoint random subsets, PCA
    each subset to num_features. Returns a list of (Ztr, Zte) pairs. The runner
    evaluates GPFN once per bag and averages the predictions.

    Unsupervised, so y is not needed and cannot leak."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[1])
    out = []
    for idx in np.array_split(order, n_bags):
        p = PCA(n_components=min(num_features, len(idx)), random_state=seed)
        ztr = p.fit_transform(x[:, idx])
        zte = p.transform(eval_x[:, idx])
        out.append((ztr, zte))
    return out


def effective_rank(z, tol=1e-8):
    """Numerical rank of a score matrix, to detect PLS saturating below 100."""
    if z.shape[1] == 0:
        return 0
    s = np.linalg.svd(np.asarray(z, dtype=np.float64), compute_uv=False)
    return int((s > (s.max() * tol)).sum()) if s.size else 0


# ---------------------------------------------------------------------------
# GPFN inference from pre-projected features
# ---------------------------------------------------------------------------

def gpfn_predict_from_features(feat_tr, train_y_norm, feat_te, model, continuous=True):
    """Run GPFN on features that have ALREADY been projected to feature_length.

    This is the post-projection block of util.inference.forward, copied verbatim
    so behaviour is identical, but with the projection lifted out so GPFN and the
    PCR control share the exact same feature array. train_y_norm must already be
    normalize()-d (the caller does it once and reuses it for PCR too).

    torch is imported lazily so this module can be sanity-checked without a GPU.
    """
    import torch

    device = next(model.parameters()).device
    bucket_means = model.bucket_means

    at = torch.as_tensor(feat_tr, device=device, dtype=torch.float32)
    bt = torch.as_tensor(feat_te, device=device, dtype=torch.float32)
    yt = torch.as_tensor(train_y_norm, device=device, dtype=torch.float32)

    x = torch.cat((at, bt), 0).unsqueeze(0).transpose(1, 0)
    y = torch.cat((yt, torch.zeros(bt.shape[0]).to(device)), 0).unsqueeze(0).transpose(1, 0)

    eval_start = at.shape[0]
    out = model(x, y, eval_start)

    if continuous:
        if hasattr(model, "loss_object"):
            pred = np.squeeze(model.loss_object.mean(out[eval_start:]).detach().cpu().numpy(), -1)
        else:
            sm = torch.nn.functional.softmax(out[eval_start:], dim=2)
            pred = np.sum(np.squeeze(sm.detach().cpu().numpy(), 1) * bucket_means, axis=1)
    else:
        pred = bucket_means[np.argmax(out[eval_start:].detach().cpu().numpy(), axis=2).flatten()]

    return pred
