#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
karst_multimodal_partial_delay_test.py
======================================
Single next-step test after the M7/M8 bridge experiment.

A) M7-MIX: keep the current seven-parameter physics, but replace the single
   unimodal population prior with a Gaussian mixture fitted to the CEM elites.

B) M9-MIX: if needed, test one minimal physics extension:
   fast bypass + partial epikarst delay. Only the non-bypass recharge is delayed.

Decision:
  M7-MIX locked AUC <= 0.75 -> KEEP_M7_MULTIMODAL
  otherwise M9-MIX <= 0.75 -> UPGRADE_TO_M9_PARTIAL_DELAY
  otherwise -> DEEPER_PHYSICS_REVISION_REQUIRED

Requires the previous file karst_domain_bridge_structural_test.py to remain in
C:/Users/franc/OneDrive/Documenti/Ricerca/Springs, plus karst_minimal_physics.py.

Spyder:
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/karst_multimodal_partial_delay_test.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

import karst_domain_bridge_structural_test as bridge
from karst_minimal_physics import (
    KarstParams, PRIOR_BOUNDS, simulate as simulate_m7, sigmoid,
    T_SNOW_C, SNOW_TRANSITION_C, T_MELT_C,
    DEGREE_DAY_FACTOR_MM_C_DAY, FAST_GATE_WIDTH_MM,
    M3_PER_MM_KM2, SECONDS_PER_DAY,
)

PARAMS7 = list(PRIOR_BOUNDS)
PARAMS9 = PARAMS7 + ["k_epi"]
EPI_BOUNDS = ("loguniform", 0.002, 0.50)
WARMUP_DAYS = bridge.WARMUP_DAYS
TOTAL_DAYS = bridge.TOTAL_DAYS
DYN_FEATURES = bridge.DYN_FEATURES


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def logit_u(u):
    u = np.clip(np.asarray(u, float), 1e-5, 1-1e-5)
    return np.log(u/(1-u))


def sigmoid_z(z):
    z = np.clip(np.asarray(z, float), -12, 12)
    return 1/(1+np.exp(-z))


def q_to_value(name, u):
    if name == "k_epi":
        fam, lo, hi = EPI_BOUNDS
    else:
        fam, lo, hi = PRIOR_BOUNDS[name]
    u = np.clip(float(u), 1e-6, 1-1e-6)
    if fam == "uniform":
        return lo + u*(hi-lo)
    if fam == "loguniform":
        return math.exp(math.log(lo) + u*(math.log(hi)-math.log(lo)))
    raise ValueError(fam)


def params7_from_u(u):
    vals = {k: q_to_value(k, u[i]) for i, k in enumerate(PARAMS7)}
    p = KarstParams(**vals)
    if p.k_fast <= p.k_slow:
        return None, vals
    return p, vals


def params9_from_u(u):
    vals = {k: q_to_value(k, u[i]) for i, k in enumerate(PARAMS9)}
    p = KarstParams(**{k: vals[k] for k in PARAMS7})
    if p.k_fast <= p.k_slow:
        return None, vals
    return p, vals


def simulate_m9(forcing, p, k_epi):
    """Selective recharge memory: direct fast bypass + delayed epikarst path."""
    idx = pd.DatetimeIndex(forcing.index)
    n = len(forcing)
    P = pd.to_numeric(forcing["precip_clean_mm"], errors="coerce").to_numpy(float)
    T = pd.to_numeric(forcing["tmean_c"], errors="coerce").to_numpy(float)
    PET = pd.to_numeric(forcing["pet_oudin_mm"], errors="coerce").to_numpy(float)

    Sslow = 0.0
    Sfast = 0.0
    SWE = 0.0
    Epi = 0.0
    q = np.full(n, np.nan)

    for i in range(n):
        if not (np.isfinite(P[i]) and np.isfinite(T[i]) and np.isfinite(PET[i])):
            continue

        snow_frac = sigmoid((T_SNOW_C - T[i]) / SNOW_TRANSITION_C)
        snowfall = max(0.0, P[i]) * snow_frac
        rainfall = max(0.0, P[i]) - snowfall
        SWE += snowfall
        potential_melt = DEGREE_DAY_FACTOR_MM_C_DAY * max(T[i] - T_MELT_C, 0.0)
        melt = min(SWE, potential_melt)
        SWE -= melt

        liquid = rainfall + melt
        R = max(0.0, liquid - p.lambda_et * max(PET[i], 0.0))

        # Same threshold meaning as M7, but applied before the delayed path.
        phi_fast = sigmoid((R - p.tau_fast_mm) / FAST_GATE_WIDTH_MM)
        I_fast = phi_fast * R

        # Only diffuse/non-bypass recharge is delayed.
        Epi += (1.0 - phi_fast) * R
        I_slow = min(Epi, k_epi * Epi)
        Epi -= I_slow

        Sfast += I_fast
        Sslow += I_slow

        qslow = min(Sslow, p.k_slow * max(Sslow, 0.0)**p.alpha_slow)
        qfast = min(Sfast, p.k_fast * max(Sfast, 0.0)**p.alpha_fast)
        Sslow -= qslow
        Sfast -= qfast

        q[i] = (
            (qslow + qfast) * p.A_eff_km2 * M3_PER_MM_KM2 / SECONDS_PER_DAY
        )

    return q


def fit_gmm(elites, param_names, kmax, seed):
    U = elites[[f"u_{k}" for k in param_names]].to_numpy(float)
    Z = logit_u(U)
    kmax = min(kmax, max(1, len(Z)//12))
    rows = []
    models = []
    for k in range(1, kmax+1):
        g = GaussianMixture(
            n_components=k, covariance_type="full", reg_covar=0.04,
            n_init=12, random_state=seed,
        )
        g.fit(Z)
        rows.append({"k": k, "bic": float(g.bic(Z)), "aic": float(g.aic(Z))})
        models.append(g)
    sel = int(np.argmin([r["bic"] for r in rows]))
    return models[sel], pd.DataFrame(rows)


def sample_gmm_u(gmm, n, rng):
    gmm.random_state = int(rng.integers(0, 2**31-1))
    z, labels = gmm.sample(n)
    return np.clip(sigmoid_z(z), .005, .995), labels


def generate_locked(model_name, gmm, fmap, locked_sites, n, seed):
    rng = np.random.default_rng(seed)
    U, labels = sample_gmm_u(gmm, n, rng)
    rows = []
    for i, u in enumerate(U):
        try:
            if model_name == "M7":
                p, vals = params7_from_u(u)
            else:
                p, vals = params9_from_u(u)
            if p is None:
                continue

            wid, forcing = bridge.choose_forcing_window(
                rng, fmap, locked_sites, TOTAL_DAYS
            )
            if model_name == "M7":
                q = simulate_m7(forcing, p)["q_phys_m3s"].to_numpy(float)
            else:
                q = simulate_m9(forcing, p, vals["k_epi"])

            sl = slice(WARMUP_DAYS, TOTAL_DAYS)
            desc = bridge.descriptors(
                forcing["precip_mm"].to_numpy(float)[sl],
                forcing["tmean_c"].to_numpy(float)[sl],
                forcing["pet_oudin_mm"].to_numpy(float)[sl],
                q[sl], forcing.index[sl],
            )
            if desc is None:
                continue
            rows.append({
                "domain": 0, "site_id": wid, "simulation": i,
                "mixture_component": int(labels[i]), **desc,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def sample_cem_u(rng, n, dim, mean=None, cov=None):
    if mean is None:
        return rng.uniform(.01, .99, size=(n, dim))
    z = rng.multivariate_normal(mean, cov, size=n)
    m = max(1, int(round(.20*n)))
    z[:m] = logit_u(rng.uniform(.01, .99, size=(m, dim)))
    return np.clip(sigmoid_z(z), .005, .995)


def eval_m9_candidate(rng, u, fmap, bridge_sites, med, iqr, nn):
    p, vals = params9_from_u(u)
    if p is None:
        return None
    try:
        wid, forcing = bridge.choose_forcing_window(
            rng, fmap, bridge_sites, TOTAL_DAYS
        )
        q = simulate_m9(forcing, p, vals["k_epi"])
        sl = slice(WARMUP_DAYS, TOTAL_DAYS)
        desc = bridge.descriptors(
            forcing["precip_mm"].to_numpy(float)[sl],
            forcing["tmean_c"].to_numpy(float)[sl],
            forcing["pet_oudin_mm"].to_numpy(float)[sl],
            q[sl], forcing.index[sl],
        )
        if desc is None:
            return None
        score = bridge.descriptor_distance(desc, med, iqr, nn)
        return {
            "site_id": wid, "score": score,
            **{f"u_{k}": float(u[i]) for i, k in enumerate(PARAMS9)},
            **desc,
        }
    except Exception:
        return None


def fit_m9_cem(bridge_real, bridge_sites, fmap, candidates, rounds,
               elite_fraction, seed, out):
    rng = np.random.default_rng(seed)
    med, iqr, Zref = bridge.robust_reference(bridge_real)
    nn = NearestNeighbors(n_neighbors=min(7, len(Zref))).fit(Zref.to_numpy(float))

    mean = None
    cov = None
    pooled = []
    history = []
    dim = len(PARAMS9)

    for rnd in range(1, rounds+1):
        print(f"\nM9 CEM round {rnd}/{rounds}")
        U = sample_cem_u(rng, candidates, dim, mean, cov)
        recs = []
        for i, u in enumerate(U):
            rec = eval_m9_candidate(rng, u, fmap, bridge_sites, med, iqr, nn)
            if rec is not None:
                rec["round"] = rnd
                recs.append(rec)
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{candidates}")

        tab = pd.DataFrame(recs)
        if len(tab) < max(30, int(.2*candidates)):
            raise RuntimeError(f"Too few valid M9 candidates: {len(tab)}")

        n_elite = max(25, int(round(len(tab)*elite_fraction)))
        elite = tab.nsmallest(n_elite, "score").copy()
        if rnd >= max(1, rounds-1):
            pooled.append(elite)

        Ze = logit_u(elite[[f"u_{k}" for k in PARAMS9]].to_numpy(float))
        nm = Ze.mean(axis=0)
        nc = np.atleast_2d(np.cov(Ze, rowvar=False)) + np.eye(dim)*0.06
        if mean is None:
            mean, cov = nm, nc
        else:
            mean = .65*mean + .35*nm
            cov = .65*cov + .35*nc

        tab.to_csv(out / f"M9_round_{rnd}_candidates.csv", index=False)
        elite.to_csv(out / f"M9_round_{rnd}_elites.csv", index=False)
        history.append({
            "round": rnd, "n_valid": int(len(tab)), "n_elite": int(len(elite)),
            "median_score": float(tab["score"].median()),
            "elite_median_score": float(elite["score"].median()),
        })
        print("  median score:", round(tab["score"].median(), 3),
              "elite:", round(elite["score"].median(), 3))

    elites = pd.concat(pooled, ignore_index=True)
    pd.DataFrame(history).to_csv(out / "M9_cem_history.csv", index=False)
    elites.to_csv(out / "M9_pooled_elites.csv", index=False)
    return elites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-dir", default="karst_inverse_study/12_domain_bridge")
    ap.add_argument("--cohort", default="karst_inverse_study/00_inventory/cohort_core_20y95.csv")
    ap.add_argument("--forcing-dir", default="karst_inverse_study/02_forcing_arco/daily")
    ap.add_argument("--out", default="karst_inverse_study/13_multimodal_partial_delay")
    ap.add_argument("--candidates", type=int, default=700)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--elite-fraction", type=float, default=.15)
    ap.add_argument("--locked-sims", type=int, default=700)
    ap.add_argument("--gmm-kmax", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    bdir = Path(args.bridge_dir)

    real_bridge = pd.read_csv(bdir / "real_bridge_descriptors.csv")
    real_locked = pd.read_csv(bdir / "real_locked_descriptors.csv")
    with open(bdir / "bridge_locked_split.json", encoding="utf-8") as f:
        split = json.load(f)
    bridge_sites = list(map(str, split["bridge_sites"]))
    locked_sites = list(map(str, split["locked_sites"]))

    cohort = pd.read_csv(args.cohort)
    fmap = bridge.build_forcing_map(cohort, args.forcing_dir)

    print("bridge sites:", len(bridge_sites))
    print("locked sites:", len(locked_sites))

    # ---- M7-MIX -----------------------------------------------------------
    print("\n" + "="*80)
    print("M7-MIX — MULTIMODAL POPULATION, SAME PHYSICS")
    print("="*80)

    m7parts = []
    for rnd in [3, 4]:
        p = bdir / "M7" / f"M7_round_{rnd}_elites.csv"
        if p.exists():
            m7parts.append(pd.read_csv(p))
    if m7parts:
        m7_elites = pd.concat(m7parts, ignore_index=True)
    else:
        m7_elites = pd.read_csv(bdir / "M7" / "final_elites.csv")

    gmm7, bic7 = fit_gmm(m7_elites, PARAMS7, args.gmm_kmax, args.seed)
    bic7.to_csv(root / "M7_gmm_selection.csv", index=False)
    syn7 = generate_locked("M7", gmm7, fmap, locked_sites,
                           args.locked_sims, args.seed+100)
    syn7.to_csv(root / "M7_locked_synthetic_descriptors.csv", index=False)
    auc7, mmd7, folds7, shifts7 = bridge.locked_domain_metrics(syn7, real_locked)
    folds7.to_csv(root / "M7_locked_domain_folds.csv", index=False)
    shifts7.to_csv(root / "M7_locked_feature_shifts.csv", index=False)
    rep7 = {
        "model": "M7-MIX", "gmm_components": int(gmm7.n_components),
        "locked_grouped_domain_auc": float(auc7),
        "locked_rbf_mmd2": float(mmd7),
        "pass_auc_le_0_75": bool(auc7 <= .75),
        "top_5_shifts": shifts7.head(5).to_dict("records"),
    }
    save_json(rep7, root / "M7_MIX_gate.json")
    print("selected GMM components:", gmm7.n_components)
    print(bic7.to_string(index=False))
    print("M7-MIX locked AUC:", round(auc7, 3))
    print("M7-MIX MMD^2:", round(mmd7, 4))
    print("PASS:", rep7["pass_auc_le_0_75"])
    print(shifts7.head(8).to_string(index=False))

    # ---- M9-MIX -----------------------------------------------------------
    print("\n" + "="*80)
    print("M9-MIX — FAST BYPASS + PARTIAL EPIKARST DELAY")
    print("="*80)
    m9dir = root / "M9"
    m9dir.mkdir(parents=True, exist_ok=True)

    m9_elites = fit_m9_cem(
        real_bridge, bridge_sites, fmap,
        args.candidates, args.rounds, args.elite_fraction,
        args.seed+1000, m9dir,
    )
    gmm9, bic9 = fit_gmm(m9_elites, PARAMS9, args.gmm_kmax, args.seed+1000)
    bic9.to_csv(root / "M9_gmm_selection.csv", index=False)
    syn9 = generate_locked("M9", gmm9, fmap, locked_sites,
                           args.locked_sims, args.seed+2000)
    syn9.to_csv(root / "M9_locked_synthetic_descriptors.csv", index=False)
    auc9, mmd9, folds9, shifts9 = bridge.locked_domain_metrics(syn9, real_locked)
    folds9.to_csv(root / "M9_locked_domain_folds.csv", index=False)
    shifts9.to_csv(root / "M9_locked_feature_shifts.csv", index=False)
    rep9 = {
        "model": "M9-MIX", "gmm_components": int(gmm9.n_components),
        "locked_grouped_domain_auc": float(auc9),
        "locked_rbf_mmd2": float(mmd9),
        "pass_auc_le_0_75": bool(auc9 <= .75),
        "physics_change": "direct fast bypass + delayed non-bypass epikarst recharge",
        "top_5_shifts": shifts9.head(5).to_dict("records"),
    }
    save_json(rep9, root / "M9_MIX_gate.json")
    print("selected GMM components:", gmm9.n_components)
    print(bic9.to_string(index=False))
    print("M9-MIX locked AUC:", round(auc9, 3))
    print("M9-MIX MMD^2:", round(mmd9, 4))
    print("PASS:", rep9["pass_auc_le_0_75"])
    print(shifts9.head(8).to_string(index=False))

    # ---- DECISION ---------------------------------------------------------
    if rep7["pass_auc_le_0_75"]:
        decision = "KEEP_M7_MULTIMODAL"
        message = (
            "Population multimodality closes the locked domain gap; keep the "
            "current seven-parameter physics and use a multimodal population prior."
        )
    elif rep9["pass_auc_le_0_75"]:
        decision = "UPGRADE_TO_M9_PARTIAL_DELAY"
        message = (
            "Multimodality alone is insufficient, but selective epikarst memory "
            "with direct fast bypass closes the locked gap. Adopt M9."
        )
    else:
        decision = "DEEPER_PHYSICS_REVISION_REQUIRED"
        message = (
            "Neither multimodal M7 nor minimally extended M9 closes the locked "
            "domain gap. A more explicit matrix-fracture-conduit exchange model is needed."
        )

    summary = {"M7_MIX": rep7, "M9_MIX": rep9,
               "decision": decision, "message": message}
    save_json(summary, root / "MULTIMODAL_PARTIAL_DELAY_DECISION.json")

    print("\n" + "="*80)
    print("FINAL DECISION")
    print("="*80)
    print("M7-MIX locked AUC:", round(auc7, 3))
    print("M9-MIX locked AUC:", round(auc9, 3))
    print("DECISION:", decision)
    print(message)


if __name__ == "__main__":
    main()
