#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
karst_support_coverage_gate.py
==============================

DECISIVE SUPPORT-COVERAGE TEST FOR APIC/SBI.

Why
---
A high synthetic-vs-real classifier AUC does NOT imply that real observations
are outside the support of the simulator. For simulation-based inference, the
critical condition is prior-predictive / simulation SUPPORT coverage, not
equality of the synthetic and real population densities.

This script compares the already-generated LOCKED simulations for:
    M7, M8, M9, M10, M11
against the same 52 LOCKED real spring sites.

No model is re-run. No CEM is re-run.

Method
------
For each candidate model:

1. Use only hydrograph-dynamic descriptors (forcing descriptors excluded).
2. Split synthetic simulations by forcing site into:
       REFERENCE sites (70%)
       CALIBRATION sites (30%)
3. Robust-standardize using REFERENCE synthetic data.
4. Fit PCA on REFERENCE and retain >=95% variance.
5. Compute k-nearest-neighbour distance to the REFERENCE cloud.
6. Set the OOD threshold as the 95th percentile of distances for
   CALIBRATION synthetic simulations.
7. Evaluate LOCKED REAL windows:
       joint_support_coverage =
           fraction(real kNN distances <= calibrated threshold)

Also report:
- site-level support coverage: fraction of real sites with at least 80% of
  their windows inside support;
- median real / calibration distance ratio;
- per-feature 99% marginal support coverage.

Decision
--------
PASS:
    joint window support >= 0.90
    site support >= 0.90

BORDERLINE:
    joint window support >= 0.80
    site support >= 0.80

FAIL:
    otherwise.

The best passing candidate should be frozen for the next APIC stage.
If none is even BORDERLINE, stop designing ad-hoc toy variants and move to a
published modular karst backbone (e.g. KarstMod-type structure).

Spyder
------
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/karst_support_coverage_gate.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


DYN_FEATURES = [
    "logq_median",
    "logq_iqr",
    "q_cv",
    "flashiness",
    "q_lag1",
    "q_lag7",
    "q_lag30",
    "zero_fraction",
    "corr_p_dlogq",
    "max_corr_p_dlogq_lag0_14",
    "seasonal_logq_amplitude",
]


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def robust_fit(X):
    med = X.median()
    iqr = X.quantile(.75) - X.quantile(.25)
    iqr[(~np.isfinite(iqr)) | (iqr < 1e-8)] = 1.0
    return med, iqr


def robust_transform(X, med, iqr):
    Z = (X - med) / iqr
    return Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def group_split_sites(syn, seed, ref_fraction=.70):
    rng = np.random.default_rng(seed)
    sites = np.array(sorted(syn["site_id"].astype(str).unique()))
    rng.shuffle(sites)

    n_ref = max(2, int(round(ref_fraction * len(sites))))
    n_ref = min(n_ref, len(sites) - 1)

    ref_sites = set(sites[:n_ref])
    cal_sites = set(sites[n_ref:])

    ref = syn[syn["site_id"].astype(str).isin(ref_sites)].copy()
    cal = syn[syn["site_id"].astype(str).isin(cal_sites)].copy()

    return ref, cal, sorted(ref_sites), sorted(cal_sites)


def pca_fit_reference(Zref, variance=.95):
    pca_full = PCA()
    pca_full.fit(Zref)

    cum = np.cumsum(pca_full.explained_variance_ratio_)
    ncomp = int(np.searchsorted(cum, variance) + 1)
    ncomp = max(2, min(ncomp, Zref.shape[1]))

    pca = PCA(n_components=ncomp)
    pca.fit(Zref)
    return pca


def knn_distance(reference, query, k=5):
    k_eff = min(k, len(reference))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(reference)
    d, _ = nn.kneighbors(query)
    return d.mean(axis=1)


def marginal_support(real, ref_syn, alpha=.01):
    rows = []
    for f in DYN_FEATURES:
        lo = float(ref_syn[f].quantile(alpha / 2))
        hi = float(ref_syn[f].quantile(1 - alpha / 2))
        x = pd.to_numeric(real[f], errors="coerce")
        ok = (x >= lo) & (x <= hi)
        rows.append({
            "feature": f,
            "synthetic_q005": lo,
            "synthetic_q995": hi,
            "real_median": float(x.median()),
            "real_marginal_coverage": float(ok.mean()),
        })
    return pd.DataFrame(rows)


def evaluate_model(
    name,
    syn,
    real,
    seed,
    k,
    threshold_quantile,
    pca_variance,
):
    ref, cal, ref_sites, cal_sites = group_split_sites(
        syn, seed
    )

    Xref = (
        ref[DYN_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )
    Xcal = (
        cal[DYN_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )
    Xreal = (
        real[DYN_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    med_fill = Xref.median()
    Xref = Xref.fillna(med_fill)
    Xcal = Xcal.fillna(med_fill)
    Xreal = Xreal.fillna(med_fill)

    rmed, riqr = robust_fit(Xref)
    Zref = robust_transform(Xref, rmed, riqr).to_numpy(float)
    Zcal = robust_transform(Xcal, rmed, riqr).to_numpy(float)
    Zreal = robust_transform(Xreal, rmed, riqr).to_numpy(float)

    pca = pca_fit_reference(Zref, pca_variance)

    Pref = pca.transform(Zref)
    Pcal = pca.transform(Zcal)
    Preal = pca.transform(Zreal)

    dcal = knn_distance(Pref, Pcal, k=k)
    dreal = knn_distance(Pref, Preal, k=k)

    threshold = float(np.quantile(dcal, threshold_quantile))

    real_eval = real[["site_id"]].copy().reset_index(drop=True)
    real_eval["support_distance"] = dreal
    real_eval["inside_joint_support"] = dreal <= threshold

    site = (
        real_eval.groupby("site_id", as_index=False)
        .agg(
            windows=("inside_joint_support", "size"),
            support_fraction=("inside_joint_support", "mean"),
            median_distance=("support_distance", "median"),
        )
    )
    site["site_supported"] = site["support_fraction"] >= .80

    joint_window_coverage = float(
        real_eval["inside_joint_support"].mean()
    )
    site_coverage = float(
        site["site_supported"].mean()
    )

    median_cal = float(np.median(dcal))
    median_real = float(np.median(dreal))
    distance_ratio = median_real / max(median_cal, 1e-12)

    marginal = marginal_support(real, ref)

    if joint_window_coverage >= .90 and site_coverage >= .90:
        status = "PASS"
    elif joint_window_coverage >= .80 and site_coverage >= .80:
        status = "BORDERLINE"
    else:
        status = "FAIL"

    report = {
        "model": name,
        "status": status,
        "n_reference_synthetic": int(len(ref)),
        "n_calibration_synthetic": int(len(cal)),
        "n_real_windows": int(len(real)),
        "n_real_sites": int(real["site_id"].nunique()),
        "n_reference_sites": len(ref_sites),
        "n_calibration_sites": len(cal_sites),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "knn_k": int(k),
        "calibration_threshold_quantile": float(threshold_quantile),
        "support_distance_threshold": threshold,
        "joint_window_support_coverage": joint_window_coverage,
        "site_support_coverage": site_coverage,
        "median_calibration_distance": median_cal,
        "median_real_distance": median_real,
        "median_real_to_calibration_distance_ratio": float(distance_ratio),
        "mean_marginal_99pct_coverage": float(
            marginal["real_marginal_coverage"].mean()
        ),
        "min_marginal_99pct_coverage": float(
            marginal["real_marginal_coverage"].min()
        ),
    }

    return report, real_eval, site, marginal


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--real",
        default=(
            "karst_inverse_study/12_domain_bridge/"
            "real_locked_descriptors.csv"
        ),
    )
    ap.add_argument(
        "--out",
        default="karst_inverse_study/15_support_coverage",
    )
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--threshold-quantile",
        type=float,
        default=.95,
    )
    ap.add_argument(
        "--pca-variance",
        type=float,
        default=.95,
    )

    args = ap.parse_args()

    candidates = {
        "M7": (
            "karst_inverse_study/12_domain_bridge/M7/"
            "locked_synthetic_descriptors.csv"
        ),
        "M8": (
            "karst_inverse_study/12_domain_bridge/M8/"
            "locked_synthetic_descriptors.csv"
        ),
        "M9": (
            "karst_inverse_study/13_multimodal_partial_delay/"
            "M9_locked_synthetic_descriptors.csv"
        ),
        "M10": (
            "karst_inverse_study/14_matrix_conduit_exchange/M10/"
            "locked_synthetic_descriptors.csv"
        ),
        "M11": (
            "karst_inverse_study/14_matrix_conduit_exchange/M11/"
            "locked_synthetic_descriptors.csv"
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(args.real)

    results = []

    for i, (name, path) in enumerate(candidates.items()):
        p = Path(path)

        if not p.exists():
            print(f"{name}: skipped — missing {p}")
            continue

        print("\n" + "=" * 78)
        print(name)
        print("=" * 78)

        syn = pd.read_csv(p)

        report, windows, sites, marginal = evaluate_model(
            name=name,
            syn=syn,
            real=real,
            seed=args.seed + 100 * i,
            k=args.k,
            threshold_quantile=args.threshold_quantile,
            pca_variance=args.pca_variance,
        )

        mdir = out / name
        mdir.mkdir(parents=True, exist_ok=True)

        save_json(report, mdir / "support_gate.json")
        windows.to_csv(
            mdir / "real_window_support.csv",
            index=False,
        )
        sites.to_csv(
            mdir / "real_site_support.csv",
            index=False,
        )
        marginal.to_csv(
            mdir / "marginal_support.csv",
            index=False,
        )

        results.append(report)

        print("status:", report["status"])
        print(
            "joint window support:",
            round(report["joint_window_support_coverage"], 3),
        )
        print(
            "site support:",
            round(report["site_support_coverage"], 3),
        )
        print(
            "median real/cal distance ratio:",
            round(
                report[
                    "median_real_to_calibration_distance_ratio"
                ],
                3,
            ),
        )
        print(
            "mean marginal 99% coverage:",
            round(report["mean_marginal_99pct_coverage"], 3),
        )

    tab = pd.DataFrame(results)

    if len(tab) == 0:
        raise RuntimeError("No candidate synthetic descriptor files found.")

    rank_order = {
        "PASS": 0,
        "BORDERLINE": 1,
        "FAIL": 2,
    }
    tab["_status_rank"] = tab["status"].map(rank_order)

    tab = tab.sort_values(
        [
            "_status_rank",
            "joint_window_support_coverage",
            "site_support_coverage",
            "median_real_to_calibration_distance_ratio",
        ],
        ascending=[True, False, False, True],
    ).drop(columns="_status_rank")

    tab.to_csv(
        out / "support_coverage_comparison.csv",
        index=False,
    )

    best = tab.iloc[0].to_dict()

    if best["status"] == "PASS":
        decision = "FREEZE_BEST_MODEL_AND_RETURN_TO_APIC"
        message = (
            f"{best['model']} provides adequate calibrated simulation support "
            "for the locked real contexts. Freeze this backbone; do not invent "
            "additional structural variants."
        )
    elif best["status"] == "BORDERLINE":
        decision = "REPEAT_SUPPORT_GATE_ACROSS_SPLITS"
        message = (
            f"{best['model']} is borderline. Repeat the support gate across "
            "3–5 bridge/locked splits before changing physics."
        )
    else:
        decision = "MOVE_TO_PUBLISHED_KARST_BACKBONE"
        message = (
            "None of M7–M11 adequately covers the locked real support. Stop "
            "adding ad-hoc variants and implement a published modular karst "
            "rainfall–discharge backbone (KarstMod-type structure)."
        )

    final = {
        "best_model": best["model"],
        "best_status": best["status"],
        "decision": decision,
        "message": message,
        "models": results,
    }

    save_json(
        final,
        out / "SUPPORT_COVERAGE_DECISION.json",
    )

    print("\n" + "=" * 78)
    print("SUPPORT-COVERAGE FINAL DECISION")
    print("=" * 78)
    print(
        tab[
            [
                "model",
                "status",
                "joint_window_support_coverage",
                "site_support_coverage",
                "median_real_to_calibration_distance_ratio",
                "mean_marginal_99pct_coverage",
            ]
        ].to_string(index=False)
    )
    print("\nBEST:", best["model"])
    print("DECISION:", decision)
    print(message)


if __name__ == "__main__":
    main()
