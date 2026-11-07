#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m11_global_practical_identifiability_confirmatory.py
=============================================

NETWORK-INDEPENDENT CONFIRMATORY POPULATION-SCALE GLOBAL PRACTICAL-IDENTIFIABILITY AUDIT FOR FROZEN M11.

Purpose
-------
Resolve the discrepancy between:
  (i) high LOCAL likelihood-informed dimension from the rescue Fisher audit;
  (ii) weak amortized recovery of most raw parameters.

The script uses only held-out synthetic S0/S1 cases and the exact M11 simulator.
No neural network is involved.

Stage A: blind multi-start exact likelihood fitting
--------------------------------------------------
For each selected TEST case and context (365, 1825 d by default):
- one oracle initialization at the true parameter vector (diagnostic only);
- Latin-hypercube blind initializations in prior-quantile space;
- bounded nonlinear least squares using the explicit synthetic log-Q noise;
- comparison of blind and oracle optima;
- search for distinct near-optimal parameter solutions;
- spread measured both in raw prior-quantile coordinates and in the
  TRAIN-defined recentered coordinates from the rescue audit.

Stage B: targeted profile likelihood
------------------------------------
A small fixed subset of cases is profiled parameter-by-parameter if Stage A
is not already decisive. At each fixed prior-quantile value, the remaining
seven parameters are reoptimized. Approximate 95% profile support uses
Delta(-2 log L) <= chi2_1(0.95) = 3.84.

Interpretation
--------------
GLOBAL_EQUIFINALITY_DETECTED:
  multiple statistically indistinguishable solutions / broad profiles exist.
  Local Fisher information is not sufficient to establish global recovery.

INFERENCE_ENGINE_FAILURE:
  blind exact optimization consistently recovers the truth / oracle solution,
  with narrow profile support. The poor amortized posterior is therefore an
  inference/training limitation rather than a fundamental information limit.

MIXED_LOCAL_GLOBAL_IDENTIFIABILITY:
  some parameters/cases are globally identifiable and others are not.

OPTIMIZATION_INCONCLUSIVE:
  blind optimization cannot reliably reach the oracle likelihood, so no
  scientific identifiability conclusion is allowed.

This script does NOT modify M11 and does NOT use real discharge.

Spyder
------
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/m11_global_practical_identifiability_confirmatory.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from scipy.stats import qmc, chi2

import m11_final_apic_stage1 as stage1
import m11_parameterisation_invariant_rescue_audit as rescue


PARAMS = list(stage1.PARAMS8)
N_PARAM = len(PARAMS)
SIGMA_LOGQ = float(stage1.LOG_Q_NOISE_SD)
CONTEXTS_DEFAULT = [365, 1825]

# Joint 95% likelihood-ratio radius for an 8-D regular model.
JOINT_DELTA_RSS_95 = float(chi2.ppf(.95, df=N_PARAM))
# 1-D 95% profile-likelihood threshold.
PROFILE_DELTA_RSS_95 = float(chi2.ppf(.95, df=1))

# -------------------------- LOCKED DECISION RULES --------------------------
# Optimization adequacy:
# median blind-vs-oracle RSS gap <= 3.84 and >=80% cases gap <= 15.51.
ORACLE_GAP_MEDIAN_MAX = PROFILE_DELTA_RSS_95
ORACLE_GAP_FRACTION_JOINT_MIN = 0.80

# Strong blind recovery if median mean absolute prior-quantile error <= .08.
STRONG_RECOVERY_MAE_U = 0.08

# A near-optimal alternative is scientifically distinct when the Euclidean
# distance in 8-D prior-quantile space exceeds .45.
DISTINCT_U_DISTANCE = 0.45

# Population evidence of global equifinality if >=25% of cases contain a
# distinct near-optimal alternative, OR targeted profiles show median 95%
# support width >= .40 for at least 3 parameters.
EQUIFINAL_CASE_FRACTION = 0.25
PROFILE_BROAD_WIDTH_U = 0.40
PROFILE_BROAD_NPARAM = 3

# If no broad profiles/alternate basins and exact blind recovery is strong,
# classify the amortized failure as inference-engine related.
# --------------------------------------------------------------------------


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def theta_from_u(u):
    u = np.clip(np.asarray(u, float), .001, .999)
    vals = np.array(
        [stage1.q_to_value(name, u[i]) for i, name in enumerate(PARAMS)],
        dtype=float,
    )
    return vals


def u_from_theta(theta):
    return np.array(
        [stage1.value_to_u(name, theta[i]) for i, name in enumerate(PARAMS)],
        dtype=float,
    )


def load_case(row):
    with np.load(row.file) as z:
        P = np.asarray(z["precip_all_mm"], dtype=np.float64)
        T = np.asarray(z["tmean_all_c"], dtype=np.float64)
        PET = np.asarray(z["pet_all_mm"], dtype=np.float64)
        qobs = np.asarray(z["q_obs_m3s"], dtype=np.float64)
        warmup = int(np.asarray(z["warmup_days"]).ravel()[0])
    return P, T, PET, qobs, warmup


def context_indices(n_exp, L):
    L = min(int(L), int(n_exp))
    st = max(0, (n_exp - L) // 2)
    return st, st + L


def make_residual_problem(P, T, PET, qobs, warmup, context_days):
    st, en = context_indices(len(qobs), context_days)
    yobs_q = qobs[st:en]
    positive = yobs_q[np.isfinite(yobs_q) & (yobs_q > 0)]
    qfloor = max(
        1e-10,
        1e-6 * float(np.median(positive)) if len(positive) else 1e-10,
    )
    yobs = np.log(np.maximum(yobs_q, qfloor))

    def residual(u):
        theta = theta_from_u(u)

        if not rescue.valid_theta(theta):
            # Fixed-length large residual. Add smooth-ish penalty for reversed
            # fast/slow ordering while keeping optimizer alive.
            kex = theta[2]
            kf = theta[4]
            p = 30.0 + 200.0 * max(0.0, kex - kf)
            return np.full(len(yobs), p, dtype=float)

        sim = rescue.simulate_from_theta(P, T, PET, theta)
        if sim is None:
            return np.full(len(yobs), 100.0, dtype=float)

        qsim = sim[0][warmup:]
        ysim = np.log(np.maximum(qsim[st:en], qfloor))
        r = (ysim - yobs) / SIGMA_LOGQ
        r[~np.isfinite(r)] = 100.0
        return r

    return residual, qfloor, st, en


def lhs_starts(n, d, seed):
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    X = sampler.random(n=n)
    return .02 + .96 * X


def fit_problem(residual, u_true, n_starts, seed, max_nfev):
    starts = [np.clip(u_true.copy(), .001, .999)]
    blind = lhs_starts(n_starts, len(u_true), seed)
    starts.extend(list(blind))

    fits = []

    for si, x0 in enumerate(starts):
        res = least_squares(
            residual,
            x0=np.asarray(x0, float),
            bounds=(np.full(N_PARAM, .001), np.full(N_PARAM, .999)),
            method="trf",
            jac="2-point",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )

        rss = float(2.0 * res.cost)
        fits.append({
            "start_index": si,
            "is_oracle_start": bool(si == 0),
            "success": bool(res.success),
            "status": int(res.status),
            "nfev": int(res.nfev),
            "rss": rss,
            "u": np.asarray(res.x, float),
            "optimality": float(res.optimality),
        })

    return fits


def recentered_z(theta, refs, coord_stats):
    c = rescue.transformed_coordinates(
        theta,
        "recentered",
        float(refs["Delta_ref_mm"]),
        float(refs["Sc_ref_mm"]),
    )

    rr = coord_stats[coord_stats.system == "recentered"].sort_values(
        "coordinate_index"
    )
    mu = rr.train_mean.to_numpy(float)
    sd = rr.train_sd.to_numpy(float)
    return (c - mu) / sd


def pairwise_max_distance(X):
    X = np.asarray(X, float)
    if len(X) < 2:
        return 0.0
    best = 0.0
    for i in range(len(X)):
        d = np.sqrt(np.sum((X[i+1:] - X[i])**2, axis=1)) if i+1 < len(X) else []
        if len(d):
            best = max(best, float(np.max(d)))
    return best


def summarize_multistart(case_id, scenario, site, context, u_true, fits, refs, coord_stats):
    oracle_fits = [f for f in fits if f["is_oracle_start"]]
    blind_fits = [f for f in fits if not f["is_oracle_start"]]

    oracle = min(oracle_fits, key=lambda f: f["rss"])
    blind_best = min(blind_fits, key=lambda f: f["rss"])
    all_best = min(fits, key=lambda f: f["rss"])

    min_rss = all_best["rss"]
    near = [f for f in fits if f["rss"] - min_rss <= JOINT_DELTA_RSS_95]

    U = np.vstack([f["u"] for f in near])
    raw_range = np.ptp(U, axis=0) if len(U) else np.full(N_PARAM, np.nan)
    raw_dist = pairwise_max_distance(U)

    Z = np.vstack([
        recentered_z(theta_from_u(f["u"]), refs, coord_stats)
        for f in near
    ]) if len(near) else np.empty((0, N_PARAM))

    z_range = np.ptp(Z, axis=0) if len(Z) else np.full(N_PARAM, np.nan)
    z_dist = pairwise_max_distance(Z)

    uhat = blind_best["u"]
    err = np.abs(uhat - u_true)

    row = {
        "case_id": case_id,
        "scenario": scenario,
        "site_id": site,
        "context_days": int(context),
        "oracle_rss": float(oracle["rss"]),
        "blind_best_rss": float(blind_best["rss"]),
        "blind_minus_oracle_rss": float(blind_best["rss"] - oracle["rss"]),
        "global_best_rss": float(min_rss),
        "blind_mean_abs_u_error": float(np.mean(err)),
        "blind_max_abs_u_error": float(np.max(err)),
        "n_nearoptimal_solutions": int(len(near)),
        "max_pairwise_nearoptimal_u_distance": float(raw_dist),
        "max_pairwise_nearoptimal_recentered_z_distance": float(z_dist),
        "has_distinct_nearoptimal_solution": bool(raw_dist >= DISTINCT_U_DISTANCE),
    }

    for j, p in enumerate(PARAMS):
        row[f"u_true_{p}"] = float(u_true[j])
        row[f"u_fit_{p}"] = float(uhat[j])
        row[f"abs_u_error_{p}"] = float(err[j])
        row[f"nearoptimal_u_range_{p}"] = float(raw_range[j])
        row[f"nearoptimal_recentered_z_range_{j+1}"] = float(z_range[j])

    run_rows = []
    for f in fits:
        rr = {
            "case_id": case_id,
            "scenario": scenario,
            "site_id": site,
            "context_days": int(context),
            "start_index": int(f["start_index"]),
            "is_oracle_start": bool(f["is_oracle_start"]),
            "success": bool(f["success"]),
            "nfev": int(f["nfev"]),
            "rss": float(f["rss"]),
            "delta_rss_from_global_best": float(f["rss"] - min_rss),
            "nearoptimal_joint95": bool(f["rss"] - min_rss <= JOINT_DELTA_RSS_95),
            "distance_from_truth_u": float(np.linalg.norm(f["u"] - u_true)),
        }
        for j, p in enumerate(PARAMS):
            rr[f"u_{p}"] = float(f["u"][j])
        run_rows.append(rr)

    return row, run_rows, blind_best


def profile_one_parameter(
    residual,
    u_mle,
    jfix,
    grid,
    max_nfev,
    seed,
):
    rng = np.random.default_rng(seed)
    free = [j for j in range(N_PARAM) if j != jfix]

    def make_full(xfree, fixed):
        u = np.empty(N_PARAM, float)
        u[jfix] = fixed
        u[free] = xfree
        return u

    rows = []

    for gi, fixed in enumerate(grid):
        starts = [u_mle[free].copy()]
        # One deterministic blind alternative makes the profile less dependent
        # on the local continuation path without exploding run time.
        starts.append(rng.uniform(.03, .97, len(free)))

        best = None

        def rfree(xfree):
            return residual(make_full(xfree, fixed))

        for x0 in starts:
            res = least_squares(
                rfree,
                x0=np.asarray(x0, float),
                bounds=(np.full(len(free), .001), np.full(len(free), .999)),
                method="trf",
                jac="2-point",
                x_scale="jac",
                max_nfev=max_nfev,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            rss = float(2.0 * res.cost)
            if best is None or rss < best[0]:
                best = (rss, make_full(res.x, fixed), int(res.nfev))

        rows.append({
            "grid_index": gi,
            "fixed_u": float(fixed),
            "rss": float(best[0]),
            "nfev_best": int(best[2]),
            "u": best[1],
        })

    minrss = min(r["rss"] for r in rows)
    for r in rows:
        r["delta_rss"] = float(r["rss"] - minrss)
        r["inside_profile95"] = bool(r["delta_rss"] <= PROFILE_DELTA_RSS_95)

    return rows


def select_profile_cases(case_summary, n_profile_cases, longest_context):
    # Fixed, transparent selection: choose cases with smallest blind-oracle gap
    # at the longest context, stratified by S0/S1 where possible. This ensures
    # profiles are not wasted on cases where optimization itself failed.
    d = case_summary[case_summary.context_days == longest_context].copy()
    d = d.sort_values(["scenario", "blind_minus_oracle_rss", "case_id"])

    pieces = []
    scenarios = [s for s in ["S0", "S1"] if s in set(d.scenario)]
    per = max(1, n_profile_cases // max(len(scenarios), 1))

    for s in scenarios:
        pieces.append(d[d.scenario == s].head(per))

    out = pd.concat(pieces, ignore_index=True) if pieces else d.head(n_profile_cases)

    if len(out) < n_profile_cases:
        used = set(out.case_id)
        extra = d[~d.case_id.isin(used)].head(n_profile_cases - len(out))
        out = pd.concat([out, extra], ignore_index=True)

    return out.head(n_profile_cases)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        default="karst_inverse_study/16_m11_final_synthetic/synthetic_manifest.csv",
    )
    ap.add_argument(
        "--rescue-cases",
        default="karst_inverse_study/24_m11_rescue_audit/rescue_audit_cases.csv",
    )
    ap.add_argument(
        "--refs",
        default="karst_inverse_study/24_m11_rescue_audit/train_reference_storages.json",
    )
    ap.add_argument(
        "--coord-stats",
        default="karst_inverse_study/24_m11_rescue_audit/coordinate_statistics.csv",
    )
    ap.add_argument(
        "--out",
        default="karst_inverse_study/26_m11_global_identifiability_confirmatory",
    )
    ap.add_argument("--n-cases", type=int, default=50)
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS_DEFAULT)
    ap.add_argument("--max-nfev", type=int, default=350)
    ap.add_argument("--profile-cases", type=int, default=6)
    ap.add_argument("--profile-grid", type=int, default=9)
    ap.add_argument("--profile-max-nfev", type=int, default=250)
    ap.add_argument("--summary-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260814)

    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.manifest)
    man = man[(man.split == "test") & man.scenario.isin(["S0", "S1"])].copy()

    if Path(args.rescue_cases).exists():
        rc = pd.read_csv(args.rescue_cases)[["case_id"]].drop_duplicates()
        man = rc.merge(man, on="case_id", how="inner")

    if len(man) < args.n_cases:
        raise RuntimeError(f"Only {len(man)} eligible TEST S0/S1 cases available.")

    # Deterministic scenario/site-aware subset from the already audited cases.
    rng = np.random.default_rng(args.seed)
    groups = list(man.groupby(["scenario", "forcing_wokas_id"]))
    rng.shuffle(groups)
    chosen = []
    for _, g in groups:
        chosen.append(g.iloc[int(rng.integers(0, len(g)))])
        if len(chosen) >= args.n_cases:
            break
    if len(chosen) < args.n_cases:
        used = {r.case_id for r in chosen}
        for r in man[~man.case_id.isin(used)].itertuples(index=False):
            chosen.append(pd.Series(r._asdict()))
            if len(chosen) >= args.n_cases:
                break
    chosen = pd.DataFrame(chosen).reset_index(drop=True)

    refs = json.loads(Path(args.refs).read_text(encoding="utf-8"))
    coord_stats = pd.read_csv(args.coord_stats)

    print("\n" + "=" * 92)
    print("M11 CONFIRMATORY GLOBAL PRACTICAL IDENTIFIABILITY AUDIT")
    print("=" * 92)
    print(f"TEST S0/S1 cases: {len(chosen)}")
    print(f"Blind starts per case/context: {args.n_starts}")
    print(f"Contexts: {list(map(int, args.contexts))}")
    print(f"Synthetic likelihood sigma_logQ: {SIGMA_LOGQ:.4f}")
    print(f"Joint near-optimal threshold DeltaRSS <= {JOINT_DELTA_RSS_95:.3f}")

    # Compile simulator.
    dummy = np.ones(32, dtype=float)
    _ = rescue.simulate_from_theta(
        dummy, dummy * 10, dummy,
        np.array([10.0, .5, .005, 1.1, .1, 1.1, 10.0, 1.0]),
    )

    case_rows = []
    run_rows = []
    best_map = {}
    problem_map = {}

    total = len(chosen) * len(args.contexts)
    done = 0

    for row in chosen.itertuples(index=False):
        P, T, PET, qobs, warmup = load_case(row)
        theta_true = np.array([float(getattr(row, f"theta_{p}")) for p in PARAMS])
        u_true = u_from_theta(theta_true)

        for context in args.contexts:
            residual, qfloor, st, en = make_residual_problem(
                P, T, PET, qobs, warmup, context
            )

            fits = fit_problem(
                residual,
                u_true,
                args.n_starts,
                args.seed + done * 97 + 13,
                args.max_nfev,
            )

            summary, runs, blind_best = summarize_multistart(
                row.case_id,
                row.scenario,
                row.forcing_wokas_id,
                context,
                u_true,
                fits,
                refs,
                coord_stats,
            )

            # Pre-specified numerical rescue: a catastrophic blind/oracle gap
            # is treated first as an optimizer failure, not as scientific
            # non-identifiability. Repeat with twice the blind starts and
            # twice the evaluation budget, then retain the combined ensemble.
            if summary["blind_minus_oracle_rss"] > JOINT_DELTA_RSS_95:
                print(
                    f"      retrying {row.case_id} {context} d: "
                    f"initial gap={summary['blind_minus_oracle_rss']:.3f}"
                )
                retry = fit_problem(
                    residual,
                    u_true,
                    args.n_starts * 2,
                    args.seed + done * 97 + 1000003,
                    args.max_nfev * 2,
                )
                # Keep one oracle trajectory and all blind trajectories from
                # both attempts. Re-index starts only for clean output.
                combined = [fits[0]] + [f for f in fits[1:]] + [f for f in retry[1:]]
                for kk, f in enumerate(combined):
                    f["start_index"] = kk
                    f["is_oracle_start"] = bool(kk == 0)

                summary, runs, blind_best = summarize_multistart(
                    row.case_id,
                    row.scenario,
                    row.forcing_wokas_id,
                    context,
                    u_true,
                    combined,
                    refs,
                    coord_stats,
                )
                summary["optimizer_retry_used"] = True
            else:
                summary["optimizer_retry_used"] = False

            case_rows.append(summary)
            run_rows.extend(runs)

            # Progressive checkpoints: useful for a long confirmatory run.
            pd.DataFrame(case_rows).to_csv(
                out / "global_case_summary_PARTIAL.csv", index=False
            )
            pd.DataFrame(run_rows).to_csv(
                out / "global_fit_runs_PARTIAL.csv", index=False
            )

            best_map[(row.case_id, int(context))] = blind_best["u"].copy()
            problem_map[(row.case_id, int(context))] = residual

            done += 1
            print(
                f"{done:3d}/{total}: {row.case_id} {context} d | "
                f"gap={summary['blind_minus_oracle_rss']:.3f} | "
                f"MAE_u={summary['blind_mean_abs_u_error']:.3f} | "
                f"near={summary['n_nearoptimal_solutions']} | "
                f"maxDist={summary['max_pairwise_nearoptimal_u_distance']:.3f}"
            )

    C = pd.DataFrame(case_rows)
    R = pd.DataFrame(run_rows)
    C.to_csv(out / "global_case_summary.csv", index=False)
    R.to_csv(out / "global_fit_runs.csv", index=False)
    chosen.to_csv(out / "global_audit_cases.csv", index=False)

    # Parameter recovery summary.
    ps = []
    for context in args.contexts:
        g = C[C.context_days == context]
        for p in PARAMS:
            e = g[f"abs_u_error_{p}"]
            rg = g[f"nearoptimal_u_range_{p}"]
            ps.append({
                "context_days": int(context),
                "parameter": p,
                "median_abs_u_error": float(e.median()),
                "p75_abs_u_error": float(e.quantile(.75)),
                "fraction_abs_u_error_le_0_10": float((e <= .10).mean()),
                "median_nearoptimal_u_range": float(rg.median()),
                "p75_nearoptimal_u_range": float(rg.quantile(.75)),
            })
    PS = pd.DataFrame(ps)
    PS.to_csv(out / "global_parameter_summary.csv", index=False)

    # Stage-A decisiveness.
    context_stage = []
    for context in args.contexts:
        g = C[C.context_days == context]
        context_stage.append({
            "context_days": int(context),
            "n_cases": int(len(g)),
            "median_blind_minus_oracle_rss": float(g.blind_minus_oracle_rss.median()),
            "fraction_blind_within_joint95_of_oracle": float(
                (g.blind_minus_oracle_rss <= JOINT_DELTA_RSS_95).mean()
            ),
            "median_blind_mean_abs_u_error": float(g.blind_mean_abs_u_error.median()),
            "fraction_cases_distinct_nearoptimal": float(
                g.has_distinct_nearoptimal_solution.mean()
            ),
            "median_max_nearoptimal_u_distance": float(
                g.max_pairwise_nearoptimal_u_distance.median()
            ),
            "median_max_nearoptimal_recentered_z_distance": float(
                g.max_pairwise_nearoptimal_recentered_z_distance.median()
            ),
            "fraction_optimizer_retry_used": float(
                g.optimizer_retry_used.mean()
            ),
        })
    ST = pd.DataFrame(context_stage)
    ST.to_csv(out / "global_context_summary.csv", index=False)

    # Population uncertainty from case-level bootstrap. These intervals quantify
    # sampling variation across held-out synthetic cases; they do not recycle
    # optimization starts as independent observations.
    brng = np.random.default_rng(args.seed + 777777)
    boot_ci_rows = []
    for context in args.contexts:
        g = C[C.context_days == context].reset_index(drop=True)
        vals = {
            "median_blind_mean_abs_u_error": [],
            "fraction_blind_within_joint95_of_oracle": [],
            "fraction_cases_distinct_nearoptimal": [],
            "median_max_nearoptimal_u_distance": [],
        }
        for _ in range(int(args.summary_bootstrap)):
            ix = brng.integers(0, len(g), size=len(g))
            b = g.iloc[ix]
            vals["median_blind_mean_abs_u_error"].append(
                float(b.blind_mean_abs_u_error.median())
            )
            vals["fraction_blind_within_joint95_of_oracle"].append(
                float((b.blind_minus_oracle_rss <= JOINT_DELTA_RSS_95).mean())
            )
            vals["fraction_cases_distinct_nearoptimal"].append(
                float(b.has_distinct_nearoptimal_solution.mean())
            )
            vals["median_max_nearoptimal_u_distance"].append(
                float(b.max_pairwise_nearoptimal_u_distance.median())
            )
        for metric, arr in vals.items():
            arr = np.asarray(arr, float)
            boot_ci_rows.append({
                "context_days": int(context),
                "metric": metric,
                "bootstrap_median": float(np.median(arr)),
                "p025": float(np.quantile(arr, .025)),
                "p975": float(np.quantile(arr, .975)),
            })
    BCI = pd.DataFrame(boot_ci_rows)
    BCI.to_csv(out / "global_context_bootstrap_CI.csv", index=False)

    longest = max(map(int, args.contexts))
    G = ST[ST.context_days == longest].iloc[0]

    optimization_ok = bool(
        G.median_blind_minus_oracle_rss <= ORACLE_GAP_MEDIAN_MAX
        and G.fraction_blind_within_joint95_of_oracle >= ORACLE_GAP_FRACTION_JOINT_MIN
    )

    stageA_equifinality = bool(
        G.fraction_cases_distinct_nearoptimal >= EQUIFINAL_CASE_FRACTION
    )

    # ------------------------------------------------------------------
    # Targeted profile likelihood. Run unless Stage A already shows clear
    # equifinality; it is useful for distinguishing narrow vs broad raw
    # parameter support when multi-start minima all collapse to one basin.
    # ------------------------------------------------------------------
    profile_rows = []
    profile_summary_rows = []

    if not stageA_equifinality and optimization_ok and args.profile_cases > 0:
        prof_cases = select_profile_cases(C, args.profile_cases, longest)
        grid = np.linspace(.05, .95, int(args.profile_grid))

        print("\nTargeted profile likelihood:")
        print(prof_cases[["case_id", "scenario", "blind_minus_oracle_rss"]].to_string(index=False))

        # Need reload problems for every profile context because only selected
        # case/context residual closures are retained in memory.
        chosen_index = chosen.set_index("case_id")

        for pr in prof_cases.itertuples(index=False):
            case_id = pr.case_id
            row = chosen_index.loc[case_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            row_obj = type("Row", (), row.to_dict())
            P, T, PET, qobs, warmup = load_case(row_obj)

            # Confirmatory profiles are evaluated only at the longest context;
            # the 365-d population conclusion comes from the full multi-start
            # audit, while profiles are reserved for the strongest-data regime.
            for context in [longest]:
                residual, _, _, _ = make_residual_problem(
                    P, T, PET, qobs, warmup, context
                )
                u_mle = best_map[(case_id, int(context))]

                for j, p in enumerate(PARAMS):
                    prof = profile_one_parameter(
                        residual,
                        u_mle,
                        j,
                        grid,
                        args.profile_max_nfev,
                        args.seed + 100000 + j * 1000 + int(context),
                    )

                    inside = [x["fixed_u"] for x in prof if x["inside_profile95"]]
                    width = float(max(inside) - min(inside)) if inside else np.nan
                    touches_edge = bool(
                        inside and (min(inside) <= grid[0] + 1e-12 or max(inside) >= grid[-1] - 1e-12)
                    )

                    profile_summary_rows.append({
                        "case_id": case_id,
                        "scenario": pr.scenario,
                        "context_days": int(context),
                        "parameter": p,
                        "profile95_width_u": width,
                        "profile95_touches_grid_edge": touches_edge,
                        "n_grid_inside95": int(len(inside)),
                    })

                    for x in prof:
                        rr = {
                            "case_id": case_id,
                            "scenario": pr.scenario,
                            "context_days": int(context),
                            "parameter": p,
                            "fixed_u": float(x["fixed_u"]),
                            "rss": float(x["rss"]),
                            "delta_rss": float(x["delta_rss"]),
                            "inside_profile95": bool(x["inside_profile95"]),
                        }
                        profile_rows.append(rr)

                    print(
                        f"  {case_id} {context} d {p}: profile95 width={width:.3f}"
                    )

        PRAW = pd.DataFrame(profile_rows)
        PSUM = pd.DataFrame(profile_summary_rows)
        PRAW.to_csv(out / "profile_likelihood_grid.csv", index=False)
        PSUM.to_csv(out / "profile_likelihood_summary.csv", index=False)
    else:
        PRAW = pd.DataFrame()
        PSUM = pd.DataFrame()

    profile_broad_params = []
    if len(PSUM):
        gl = PSUM[PSUM.context_days == longest]
        medprof = (
            gl.groupby("parameter", as_index=False)
            .agg(
                median_profile95_width_u=("profile95_width_u", "median"),
                fraction_touching_edge=("profile95_touches_grid_edge", "mean"),
            )
        )
        medprof.to_csv(out / "profile_parameter_summary.csv", index=False)
        profile_broad_params = medprof.loc[
            medprof.median_profile95_width_u >= PROFILE_BROAD_WIDTH_U,
            "parameter",
        ].tolist()

    profile_equifinality = bool(
        len(profile_broad_params) >= PROFILE_BROAD_NPARAM
    )

    strong_exact_recovery = bool(
        G.median_blind_mean_abs_u_error <= STRONG_RECOVERY_MAE_U
    )

    if not optimization_ok:
        verdict = "OPTIMIZATION_INCONCLUSIVE"
        action = (
            "Blind exact optimization does not reach the oracle likelihood reliably. "
            "Do not attribute the APIC discrepancy to physics or inference yet."
        )
    elif stageA_equifinality or profile_equifinality:
        verdict = "GLOBAL_EQUIFINALITY_DETECTED"
        action = (
            "The exact M11 likelihood admits globally broad or distinct near-optimal "
            "parameter solutions despite high local Fisher information. Reframe the "
            "paper around the distinction between local observability and global "
            "practical identifiability; do not use local Fisher dimension as a raw-"
            "parameter recovery claim."
        )
    elif strong_exact_recovery:
        verdict = "INFERENCE_ENGINE_FAILURE"
        action = (
            "Exact blind likelihood optimization recovers the held-out synthetic "
            "parameters without evidence of broad global alternatives. The poor "
            "amortized raw-parameter posterior is therefore primarily an inference-"
            "engine/training limitation. Do not claim fundamental non-identifiability."
        )
    else:
        verdict = "MIXED_LOCAL_GLOBAL_IDENTIFIABILITY"
        action = (
            "The evidence is mixed: exact optimization is adequate, but recovery is "
            "not uniformly strong and broad global alternatives are not dominant. "
            "Treat identifiability as parameter- and site-dependent and avoid a single "
            "global rank claim."
        )

    report = {
        "verdict": verdict,
        "scientific_action": action,
        "n_cases": int(len(chosen)),
        "contexts_days": list(map(int, args.contexts)),
        "blind_starts_per_case_context": int(args.n_starts),
        "sigma_logQ": SIGMA_LOGQ,
        "joint_nearoptimal_deltaRSS95": JOINT_DELTA_RSS_95,
        "profile_deltaRSS95": PROFILE_DELTA_RSS_95,
        "longest_context_summary": {
            k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
            for k, v in G.to_dict().items()
        },
        "optimization_ok": optimization_ok,
        "stageA_equifinality": stageA_equifinality,
        "profile_equifinality": profile_equifinality,
        "profile_broad_parameters": profile_broad_params,
        "strong_exact_recovery": strong_exact_recovery,
        "population_bootstrap_CI_file": "global_context_bootstrap_CI.csv",
        "optimizer_retry_rule": (
            "If initial blind-oracle gap exceeds joint 95% LR threshold, repeat "
            "with 2x blind starts and 2x max_nfev before classification."
        ),
        "locked_thresholds": {
            "ORACLE_GAP_MEDIAN_MAX": ORACLE_GAP_MEDIAN_MAX,
            "ORACLE_GAP_FRACTION_JOINT_MIN": ORACLE_GAP_FRACTION_JOINT_MIN,
            "STRONG_RECOVERY_MAE_U": STRONG_RECOVERY_MAE_U,
            "DISTINCT_U_DISTANCE": DISTINCT_U_DISTANCE,
            "EQUIFINAL_CASE_FRACTION": EQUIFINAL_CASE_FRACTION,
            "PROFILE_BROAD_WIDTH_U": PROFILE_BROAD_WIDTH_U,
            "PROFILE_BROAD_NPARAM": PROFILE_BROAD_NPARAM,
        },
    }

    save_json(report, out / "GLOBAL_IDENTIFIABILITY_VERDICT.json")

    print("\n" + "=" * 92)
    print("GLOBAL IDENTIFIABILITY — CONTEXT SUMMARY")
    print("=" * 92)
    print(ST.to_string(index=False))

    print("\nPopulation bootstrap 95% intervals:")
    print(BCI.to_string(index=False))

    print("\nParameter recovery summary:")
    print(PS[PS.context_days == longest].to_string(index=False))

    if len(PSUM):
        print("\nProfile likelihood summary at longest context:")
        print(
            PSUM[PSUM.context_days == longest]
            .groupby("parameter", as_index=False)
            .agg(
                median_profile95_width_u=("profile95_width_u", "median"),
                fraction_touching_edge=("profile95_touches_grid_edge", "mean"),
            )
            .to_string(index=False)
        )

    print("\n" + "=" * 92)
    print("VERDICT:", verdict)
    print("=" * 92)
    print(action)
    print("\nMachine-readable verdict:", out / "GLOBAL_IDENTIFIABILITY_VERDICT.json")


if __name__ == "__main__":
    main()
