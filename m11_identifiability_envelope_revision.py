#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m11_identifiability_envelope_revision.py
=======================================

ONE-SHOT POST-PROCESSING FOR THE JoH REVISION.

No M11 simulations are rerun.
No new model is fitted.
No S2/S3 analysis is performed.

This script addresses the high-value methodological points raised in review:
1) noise-dependent identifiability envelope d_inf(L, sigma_logQ);
2) complete Fisher eigenvalue spectrum;
3) forcing-site cluster bootstrap for the 50-case confirmatory audit;
4) parameter recovery errors in native units;
5) sensitivity of "distinct near-optimal" fractions to distance threshold;
6) real-data spin-up distribution and its association with fit diagnostics;
7) fixed global M11 constants exported explicitly for the manuscript.

Expected existing outputs
-------------------------
karst_inverse_study/24_m11_rescue_audit/case_fisher_matrices.npz
karst_inverse_study/26_m11_global_identifiability_confirmatory/global_case_summary.csv
karst_inverse_study/27_m11_real_exact_inversion_v2/real_exact_primary_fits.csv
karst_inverse_study/27_m11_real_exact_inversion_v2/real_selected_windows.csv

Output
------
karst_inverse_study/29_m11_identifiability_envelope/
    IDENTIFIABILITY_ENVELOPE_SUMMARY.json
    identifiability_envelope.csv
    fisher_eigenvalue_spectrum.csv
    envelope_landmarks.csv
    cluster_bootstrap_summary.csv
    native_parameter_recovery.csv
    nearoptimal_threshold_sensitivity.csv
    real_spinup_summary.csv
    fixed_M11_constants.csv
    figures/
        Figure_identifiability_envelope.png
        Figure_fisher_spectrum.png
        Figure_threshold_sensitivity.png
        Figure_real_spinup.png

Interpretation
--------------
The real fitted log-RMSE distribution is shown on the envelope figure ONLY
as an empirical model-data residual scale. It is NOT treated as an estimate
of pure discharge measurement error.

Spyder
------
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/m11_identifiability_envelope_revision.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import m11_final_apic_stage1 as stage1
import karst_minimal_physics as phys


BASE_SIGMA = float(stage1.LOG_Q_NOISE_SD)
PARAMS = list(stage1.PARAMS8)
N_PARAM = len(PARAMS)

DISTANCE_THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.80]
LANDMARK_SIGMAS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def save_json(obj, path):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_float(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else None
    except Exception:
        return None


def fisher_eigenvalues(F):
    F = np.asarray(F, dtype=float)
    F = 0.5 * (F + F.T)
    lam = np.linalg.eigvalsh(F)
    return np.clip(np.sort(lam)[::-1], 0.0, None)


def d_inf_from_lambda(lam):
    lam = np.asarray(lam, dtype=float)
    return float(np.sum(lam / (1.0 + lam)))


def scale_lambda(lam_base, sigma):
    return np.asarray(lam_base, float) * (BASE_SIGMA / float(sigma)) ** 2


def quantile_row(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(median=np.nan, p25=np.nan, p75=np.nan, p025=np.nan, p975=np.nan)
    return dict(
        median=float(np.median(x)),
        p25=float(np.quantile(x, .25)),
        p75=float(np.quantile(x, .75)),
        p025=float(np.quantile(x, .025)),
        p975=float(np.quantile(x, .975)),
    )


def infer_contexts(npz):
    out = []
    for k in npz.files:
        m = re.fullmatch(r"original_(\d+)d", k)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def envelope_and_spectrum(fisher_path, sigma_min, sigma_max, n_sigma):
    dat = np.load(fisher_path)
    contexts = infer_contexts(dat)
    if not contexts:
        raise RuntimeError(
            f"No keys like original_365d found in {fisher_path}; keys={dat.files}"
        )

    sigma_grid = np.geomspace(sigma_min, sigma_max, n_sigma)

    envelope_rows = []
    spectrum_rows = []
    landmark_rows = []

    base_case_lambdas = {}

    for L in contexts:
        Fcases = np.asarray(dat[f"original_{L}d"], dtype=float)
        if Fcases.ndim != 3 or Fcases.shape[1:] != (N_PARAM, N_PARAM):
            raise RuntimeError(f"Unexpected Fisher shape for {L} d: {Fcases.shape}")

        LAMS = np.vstack([fisher_eigenvalues(F) for F in Fcases])
        base_case_lambdas[L] = LAMS

        # Full case-wise spectrum at baseline sigma.
        for direction in range(N_PARAM):
            q = quantile_row(LAMS[:, direction])
            spectrum_rows.append({
                "context_days": L,
                "direction": direction + 1,
                "sigma_logQ": BASE_SIGMA,
                "median_eigenvalue": q["median"],
                "p25_eigenvalue": q["p25"],
                "p75_eigenvalue": q["p75"],
                "p025_eigenvalue": q["p025"],
                "p975_eigenvalue": q["p975"],
                "median_contraction": float(
                    np.median(
                        LAMS[:, direction] / (1.0 + LAMS[:, direction])
                    )
                ),
            })

        # Noise-dependent informed dimension.
        for sigma in sigma_grid:
            scale = (BASE_SIGMA / sigma) ** 2
            lam_s = LAMS * scale
            dinf = np.sum(lam_s / (1.0 + lam_s), axis=1)
            q = quantile_row(dinf)
            envelope_rows.append({
                "context_days": L,
                "sigma_logQ": float(sigma),
                "median_d_informed": q["median"],
                "p25_d_informed": q["p25"],
                "p75_d_informed": q["p75"],
                "p025_d_informed": q["p025"],
                "p975_d_informed": q["p975"],
                "median_n_lambda_gt1": float(np.median(np.sum(lam_s > 1.0, axis=1))),
                "median_n_lambda_gt9": float(np.median(np.sum(lam_s > 9.0, axis=1))),
            })

        for sigma in LANDMARK_SIGMAS:
            scale = (BASE_SIGMA / sigma) ** 2
            lam_s = LAMS * scale
            dinf = np.sum(lam_s / (1.0 + lam_s), axis=1)
            q = quantile_row(dinf)
            landmark_rows.append({
                "context_days": L,
                "sigma_logQ": sigma,
                "median_d_informed": q["median"],
                "p25_d_informed": q["p25"],
                "p75_d_informed": q["p75"],
                "fraction_d_informed_ge_6": float(np.mean(dinf >= 6.0)),
                "fraction_d_informed_ge_4": float(np.mean(dinf >= 4.0)),
                "fraction_d_informed_ge_2": float(np.mean(dinf >= 2.0)),
            })

    E = pd.DataFrame(envelope_rows)
    S = pd.DataFrame(spectrum_rows)
    LM = pd.DataFrame(landmark_rows)

    # Where does the median curve cross integer informed dimensions?
    crossings = []
    targets = [7, 6, 5, 4, 3, 2, 1]
    for L, g in E.groupby("context_days"):
        g = g.sort_values("sigma_logQ")
        x = g["sigma_logQ"].to_numpy(float)
        y = g["median_d_informed"].to_numpy(float)
        for target in targets:
            # first sigma where median falls below target
            idx = np.where(y <= target)[0]
            sigma_cross = float(x[idx[0]]) if len(idx) else np.nan
            crossings.append({
                "context_days": int(L),
                "target_median_d_informed": target,
                "first_sigma_at_or_below_target": sigma_cross,
            })
    X = pd.DataFrame(crossings)

    return E, S, LM, X, base_case_lambdas


def cluster_bootstrap(confirmatory_case_path, n_boot, seed):
    C = pd.read_csv(confirmatory_case_path)
    if "site_id" not in C.columns:
        raise RuntimeError("global_case_summary.csv lacks site_id")

    rows = []
    rng = np.random.default_rng(seed)

    for context, g0 in C.groupby("context_days"):
        g0 = g0.copy()
        sites = g0["site_id"].astype(str).drop_duplicates().to_numpy()
        if len(sites) < 2:
            continue

        boots = {
            "median_blind_mean_abs_u_error": [],
            "fraction_blind_within_joint95_of_oracle": [],
            "fraction_cases_distinct_nearoptimal": [],
            "median_max_nearoptimal_u_distance": [],
        }

        # Threshold used by the confirmatory script.
        joint95 = 15.507

        groups = {s: g0[g0.site_id.astype(str) == s] for s in sites}

        for _ in range(n_boot):
            sampled = rng.choice(sites, size=len(sites), replace=True)
            pieces = []
            for draw_i, s in enumerate(sampled):
                z = groups[s].copy()
                # preserve multiplicity of resampled clusters
                z["_cluster_draw"] = draw_i
                pieces.append(z)
            b = pd.concat(pieces, ignore_index=True)

            boots["median_blind_mean_abs_u_error"].append(
                float(b.blind_mean_abs_u_error.median())
            )
            boots["fraction_blind_within_joint95_of_oracle"].append(
                float((b.blind_minus_oracle_rss <= joint95).mean())
            )
            boots["fraction_cases_distinct_nearoptimal"].append(
                float(b.has_distinct_nearoptimal_solution.astype(bool).mean())
            )
            boots["median_max_nearoptimal_u_distance"].append(
                float(b.max_pairwise_nearoptimal_u_distance.median())
            )

        for metric, vals in boots.items():
            q = quantile_row(vals)
            observed = {
                "median_blind_mean_abs_u_error":
                    float(g0.blind_mean_abs_u_error.median()),
                "fraction_blind_within_joint95_of_oracle":
                    float((g0.blind_minus_oracle_rss <= joint95).mean()),
                "fraction_cases_distinct_nearoptimal":
                    float(g0.has_distinct_nearoptimal_solution.astype(bool).mean()),
                "median_max_nearoptimal_u_distance":
                    float(g0.max_pairwise_nearoptimal_u_distance.median()),
            }[metric]
            rows.append({
                "context_days": int(context),
                "metric": metric,
                "observed": observed,
                "cluster_bootstrap_median": q["median"],
                "cluster_bootstrap_p025": q["p025"],
                "cluster_bootstrap_p975": q["p975"],
                "n_forcing_sites": int(len(sites)),
                "n_cases": int(len(g0)),
                "n_boot": int(n_boot),
            })

    return pd.DataFrame(rows)


def native_parameter_recovery(confirmatory_case_path):
    C = pd.read_csv(confirmatory_case_path)
    rows = []

    for context, g in C.groupby("context_days"):
        for p in PARAMS:
            ut = pd.to_numeric(g[f"u_true_{p}"], errors="coerce").to_numpy(float)
            uf = pd.to_numeric(g[f"u_fit_{p}"], errors="coerce").to_numpy(float)

            true = np.array([stage1.q_to_value(p, u) for u in ut], float)
            fit = np.array([stage1.q_to_value(p, u) for u in uf], float)

            ae = np.abs(fit - true)
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = 100.0 * ae / np.abs(true)
            ape[~np.isfinite(ape)] = np.nan

            rows.append({
                "context_days": int(context),
                "parameter": p,
                "n_cases": int(len(g)),
                "median_abs_u_error": float(
                    np.nanmedian(np.abs(uf - ut))
                ),
                "median_abs_native_error": float(np.nanmedian(ae)),
                "p75_abs_native_error": float(np.nanquantile(ae, .75)),
                "median_abs_percent_error": float(np.nanmedian(ape)),
                "p75_abs_percent_error": float(np.nanquantile(ape, .75)),
                "native_unit": {
                    "A_eff_km2": "km2",
                    "lambda_et": "dimensionless",
                    "k_slow": "model coefficient",
                    "alpha_slow": "dimensionless",
                    "k_fast": "model coefficient",
                    "alpha_fast": "dimensionless",
                    "tau_fast_mm": "mm",
                    "rho_mc": "dimensionless",
                }.get(p, ""),
            })

    return pd.DataFrame(rows)


def threshold_sensitivity(confirmatory_case_path, real_fit_path):
    C = pd.read_csv(confirmatory_case_path)
    R = pd.read_csv(real_fit_path)
    R = R[R["status"] == "ok"].copy()

    rows = []
    for context, g in C.groupby("context_days"):
        x = pd.to_numeric(
            g.max_pairwise_nearoptimal_u_distance, errors="coerce"
        ).dropna()
        for th in DISTANCE_THRESHOLDS:
            rows.append({
                "dataset": "matched_synthetic",
                "context_days": int(context),
                "threshold_u_distance": th,
                "n_cases": int(len(x)),
                "fraction_above_threshold": float((x >= th).mean()),
            })

    x = pd.to_numeric(
        R.max_nearoptimal_u_distance, errors="coerce"
    ).dropna()
    for th in DISTANCE_THRESHOLDS:
        rows.append({
            "dataset": "real_calibration",
            "context_days": 1825,
            "threshold_u_distance": th,
            "n_cases": int(len(x)),
            "fraction_above_threshold": float((x >= th).mean()),
        })

    return pd.DataFrame(rows)


def real_spinup_analysis(real_fit_path, real_window_path):
    F = pd.read_csv(real_fit_path)
    F = F[F["status"] == "ok"].copy()
    W = pd.read_csv(real_window_path)
    D = F.merge(
        W[["wokas_id", "warmup_days_used"]].drop_duplicates("wokas_id"),
        on="wokas_id",
        how="left",
        suffixes=("", "_window"),
    )

    # Prefer the window-file value if duplicate.
    if "warmup_days_used_window" in D.columns:
        D["warmup_days"] = pd.to_numeric(
            D["warmup_days_used_window"], errors="coerce"
        )
    else:
        D["warmup_days"] = pd.to_numeric(
            D["warmup_days_used"], errors="coerce"
        )

    w = D["warmup_days"].dropna()
    rows = [{
        "metric": "warmup_days_used",
        "n_sites": int(len(w)),
        "min": float(w.min()) if len(w) else np.nan,
        "p25": float(w.quantile(.25)) if len(w) else np.nan,
        "median": float(w.median()) if len(w) else np.nan,
        "p75": float(w.quantile(.75)) if len(w) else np.nan,
        "max": float(w.max()) if len(w) else np.nan,
        "fraction_ge_3650": float((w >= 3650).mean()) if len(w) else np.nan,
        "fraction_lt_1825": float((w < 1825).mean()) if len(w) else np.nan,
    }]

    for metric in [
        "rmse_logQ", "KGE", "logNSE",
        "residual_acf1", "residual_acf7", "residual_acf30",
    ]:
        if metric not in D.columns:
            continue
        x = pd.to_numeric(D["warmup_days"], errors="coerce")
        y = pd.to_numeric(D[metric], errors="coerce")
        ok = np.isfinite(x) & np.isfinite(y)
        if (
            ok.sum() >= 5
            and np.nanstd(x[ok]) > 0
            and np.nanstd(y[ok]) > 0
        ):
            rho, pval = spearmanr(x[ok], y[ok])
        else:
            rho, pval = np.nan, np.nan
        rows.append({
            "metric": f"spearman_warmup_vs_{metric}",
            "n_sites": int(ok.sum()),
            "rho": safe_float(rho),
            "p_value": safe_float(pval),
        })

    return pd.DataFrame(rows), D


def fixed_constants_table():
    rows = [
        ("snow_temperature_threshold", float(phys.T_SNOW_C), "degC"),
        ("snow_transition_width", float(phys.SNOW_TRANSITION_C), "degC"),
        ("melt_temperature_threshold", float(phys.T_MELT_C), "degC"),
        (
            "degree_day_melt_factor",
            float(phys.DEGREE_DAY_FACTOR_MM_C_DAY),
            "mm degC^-1 d^-1",
        ),
        ("fast_gate_width", float(phys.FAST_GATE_WIDTH_MM), "mm"),
        ("synthetic_logQ_noise_sd", BASE_SIGMA, "log discharge"),
        ("synthetic_warmup_days", int(stage1.WARMUP_DAYS), "d"),
    ]
    return pd.DataFrame(rows, columns=["constant", "value", "unit"])


def make_figures(E, S, TS, spin_df, real_fit_path, outfig):
    outfig.mkdir(parents=True, exist_ok=True)

    R = pd.read_csv(real_fit_path)
    R = R[R["status"] == "ok"].copy()
    real_rmse = pd.to_numeric(R["rmse_logQ"], errors="coerce").dropna()

    # Figure 1: identifiability envelope.
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for L, g in E.groupby("context_days"):
        g = g.sort_values("sigma_logQ")
        ax.plot(
            g["sigma_logQ"],
            g["median_d_informed"],
            label=f"{int(L)} d",
        )
        ax.fill_between(
            g["sigma_logQ"].to_numpy(float),
            g["p25_d_informed"].to_numpy(float),
            g["p75_d_informed"].to_numpy(float),
            alpha=0.12,
        )

    ax.axvline(BASE_SIGMA, linestyle="--", linewidth=1.0)
    if len(real_rmse):
        med = float(real_rmse.median())
        q25 = float(real_rmse.quantile(.25))
        q75 = float(real_rmse.quantile(.75))
        ax.axvline(med, linestyle=":", linewidth=1.2)
        ax.axvspan(q25, q75, alpha=0.08)

    ax.set_xscale("log")
    ax.set_xlim(E.sigma_logQ.min(), E.sigma_logQ.max())
    ax.set_ylim(0, N_PARAM + 0.15)
    ax.set_xlabel(r"Assumed independent log-discharge error, $\sigma_{\log Q}$")
    ax.set_ylabel(r"Likelihood-informed dimension, $d_{\mathrm{inf}}$")
    ax.legend(frameon=False, title="Record length")
    fig.tight_layout()
    fig.savefig(outfig / "Figure_identifiability_envelope.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Fisher spectrum at baseline sigma.
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for L, g in S.groupby("context_days"):
        g = g.sort_values("direction")
        ax.plot(
            g["direction"],
            g["median_eigenvalue"],
            marker="o",
            label=f"{int(L)} d",
        )
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.axhline(9.0, linestyle=":", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(range(1, N_PARAM + 1))
    ax.set_xlabel("Fisher eigen-direction")
    ax.set_ylabel(r"Median eigenvalue at $\sigma_{\log Q}=0.02$")
    ax.legend(frameon=False, title="Record length")
    fig.tight_layout()
    fig.savefig(outfig / "Figure_fisher_spectrum.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: threshold sensitivity.
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for (dataset, context), g in TS.groupby(["dataset", "context_days"]):
        lab = (
            f"Synthetic {int(context)} d"
            if dataset == "matched_synthetic"
            else "Real calibration 1825 d"
        )
        ax.plot(
            g["threshold_u_distance"],
            g["fraction_above_threshold"],
            marker="o",
            label=lab,
        )
    ax.axvline(.45, linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Distance threshold in prior-quantile space")
    ax.set_ylabel("Fraction with a distant near-optimal solution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfig / "Figure_threshold_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: real spin-up vs RMSE if variable.
    if len(spin_df):
        x = pd.to_numeric(spin_df["warmup_days"], errors="coerce")
        y = pd.to_numeric(spin_df["rmse_logQ"], errors="coerce")
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum():
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            ax.scatter(x[ok], y[ok])
            ax.set_xlabel("Forcing-only spin-up used (days)")
            ax.set_ylabel("Real calibration log-RMSE")
            fig.tight_layout()
            fig.savefig(outfig / "Figure_real_spinup.png", dpi=600, bbox_inches="tight")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fisher",
        default="karst_inverse_study/24_m11_rescue_audit/case_fisher_matrices.npz",
    )
    ap.add_argument(
        "--confirmatory-cases",
        default=(
            "karst_inverse_study/26_m11_global_identifiability_confirmatory/"
            "global_case_summary.csv"
        ),
    )
    ap.add_argument(
        "--real-fits",
        default=(
            "karst_inverse_study/27_m11_real_exact_inversion_v2/"
            "real_exact_primary_fits.csv"
        ),
    )
    ap.add_argument(
        "--real-windows",
        default=(
            "karst_inverse_study/27_m11_real_exact_inversion_v2/"
            "real_selected_windows.csv"
        ),
    )
    ap.add_argument(
        "--out",
        default="karst_inverse_study/29_m11_identifiability_envelope",
    )
    ap.add_argument("--sigma-min", type=float, default=.01)
    ap.add_argument("--sigma-max", type=float, default=.50)
    ap.add_argument("--n-sigma", type=int, default=240)
    ap.add_argument("--cluster-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()

    required = [
        Path(args.fisher),
        Path(args.confirmatory_cases),
        Path(args.real_fits),
        Path(args.real_windows),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required existing outputs not found:\n  " + "\n  ".join(missing)
        )

    out = Path(args.out)
    figdir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 92)
    print("M11 IDENTIFIABILITY ENVELOPE — JoH REVISION POST-PROCESSING")
    print("=" * 92)
    print("No new M11 simulations or calibrations are performed.")

    E, S, LM, X, _ = envelope_and_spectrum(
        args.fisher,
        args.sigma_min,
        args.sigma_max,
        args.n_sigma,
    )
    E.to_csv(out / "identifiability_envelope.csv", index=False)
    S.to_csv(out / "fisher_eigenvalue_spectrum.csv", index=False)
    LM.to_csv(out / "envelope_landmarks.csv", index=False)
    X.to_csv(out / "envelope_crossings.csv", index=False)

    CB = cluster_bootstrap(
        args.confirmatory_cases,
        args.cluster_bootstrap,
        args.seed,
    )
    CB.to_csv(out / "cluster_bootstrap_summary.csv", index=False)

    NR = native_parameter_recovery(args.confirmatory_cases)
    NR.to_csv(out / "native_parameter_recovery.csv", index=False)

    TS = threshold_sensitivity(args.confirmatory_cases, args.real_fits)
    TS.to_csv(out / "nearoptimal_threshold_sensitivity.csv", index=False)

    SP, spin_df = real_spinup_analysis(args.real_fits, args.real_windows)
    SP.to_csv(out / "real_spinup_summary.csv", index=False)

    FC = fixed_constants_table()
    FC.to_csv(out / "fixed_M11_constants.csv", index=False)

    make_figures(E, S, TS, spin_df, args.real_fits, figdir)

    R = pd.read_csv(args.real_fits)
    R = R[R["status"] == "ok"].copy()
    rmse = pd.to_numeric(R["rmse_logQ"], errors="coerce").dropna()

    summary = {
        "analysis_type": "post_processing_only",
        "no_new_M11_simulation": True,
        "no_new_calibration": True,
        "baseline_sigma_logQ": BASE_SIGMA,
        "sigma_envelope_range": [args.sigma_min, args.sigma_max],
        "contexts_days": sorted(E.context_days.unique().astype(int).tolist()),
        "real_rmse_is_model_data_residual_scale_not_measurement_error": True,
        "real_logRMSE_median": float(rmse.median()) if len(rmse) else None,
        "real_logRMSE_p25": float(rmse.quantile(.25)) if len(rmse) else None,
        "real_logRMSE_p75": float(rmse.quantile(.75)) if len(rmse) else None,
        "fixed_M11_constants": {
            r["constant"]: r["value"] for _, r in FC.iterrows()
        },
    }

    # Compact landmark dictionary for direct manuscript use.
    summary["d_informed_landmarks"] = {}
    for L in sorted(LM.context_days.unique()):
        z = LM[LM.context_days == L]
        summary["d_informed_landmarks"][str(int(L))] = {
            str(float(r.sigma_logQ)): {
                "median": float(r.median_d_informed),
                "p25": float(r.p25_d_informed),
                "p75": float(r.p75_d_informed),
            }
            for _, r in z.iterrows()
        }

    save_json(summary, out / "IDENTIFIABILITY_ENVELOPE_SUMMARY.json")

    print("\nIDENTIFIABILITY ENVELOPE — LANDMARKS")
    print(LM.to_string(index=False))

    print("\nFISHER EIGENVALUE SPECTRUM AT sigma_logQ=0.02")
    print(S.to_string(index=False))

    print("\nFORCING-SITE CLUSTER BOOTSTRAP")
    print(CB.to_string(index=False))

    print("\nNATIVE PARAMETER RECOVERY")
    print(NR.to_string(index=False))

    print("\nNEAR-OPTIMAL DISTANCE THRESHOLD SENSITIVITY")
    print(TS.to_string(index=False))

    print("\nREAL SPIN-UP CHECK")
    print(SP.to_string(index=False))

    print("\nFIXED M11 CONSTANTS")
    print(FC.to_string(index=False))

    print("\n" + "=" * 92)
    print("POST-PROCESSING COMPLETE")
    print("=" * 92)
    print(f"Outputs: {out}")
    print(
        "Paste the console blocks above (or upload IDENTIFIABILITY_ENVELOPE_SUMMARY.json "
        "plus the CSV tables) to update the manuscript."
    )


if __name__ == "__main__":
    main()
