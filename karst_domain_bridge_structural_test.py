#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
karst_domain_bridge_structural_test.py
======================================

UNICO script per risolvere il domain-gap synthetic -> real prima di nuove
repliche APIC.

Domanda scientifica
-------------------
Il gap osservato deriva principalmente da:
A) population prior troppo ampio / mal centrato sul modello fisico minimale
   attuale a 7 parametri?
oppure
B) mancanza strutturale di una semplice memoria di ricarica?

Strategia rigorosa
------------------
1) Split REALI CORE per sito, stratificato per paese:
      BRIDGE ~ 1/3 dei siti
      LOCKED ~ 2/3 dei siti
   Il LOCKED set non viene usato per apprendere la population prior.

2) Costruisce descrittori idrologici 365-d dai soli siti BRIDGE.

3) Usa Cross-Entropy Method (CEM) per apprendere una population prior in
   prior-quantile space, confrontando simulazioni con i descrittori BRIDGE.

4) Esegue due esperimenti separati:
   M7 = modello minimale corrente, 7 parametri;
   M8 = stesso modello + UN SOLO recharge-delay reservoir globale/site-specifico
        con parametro k_delay.

5) Valuta entrambi ESCLUSIVAMENTE sui siti LOCKED:
   - grouped domain-classifier AUC;
   - MMD;
   - feature shifts.

Decisione:
   - se M7 AUC <= 0.75: tenere il modello 7-parametri;
   - se M7 fallisce ma M8 AUC <= 0.75: aggiungere recharge delay;
   - se entrambi falliscono: non cambiare APIC; serve revisione strutturale
     più profonda del simulatore fisico.

Nessuna portata reale viene usata come target di calibrazione puntuale.
Il bridge usa SOLO distribuzioni aggregate di descrittori dinamici.

Input attesi
------------
karst_inverse_study/00_inventory/cohort_core_20y95.csv
karst_inverse_study/01_discharge/*.csv
karst_inverse_study/02_forcing_arco/daily/*.csv
karst_minimal_physics.py

Output
------
karst_inverse_study/12_domain_bridge/

Esecuzione Spyder
-----------------
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/karst_domain_bridge_structural_test.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"

Smoke test:
%runfile ".../karst_domain_bridge_structural_test.py" --args "--candidates 250 --rounds 2 --locked-sims 250" --wdir "..."
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from karst_minimal_physics import (
    KarstParams,
    PRIOR_BOUNDS,
    simulate as simulate_m7,
    sigmoid,
    T_SNOW_C,
    SNOW_TRANSITION_C,
    T_MELT_C,
    DEGREE_DAY_FACTOR_MM_C_DAY,
    FAST_GATE_WIDTH_MM,
    M3_PER_MM_KM2,
    SECONDS_PER_DAY,
)

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

CONTEXT_DAYS = 365
WARMUP_DAYS = 10 * 365 + 3
TOTAL_DAYS = WARMUP_DAYS + CONTEXT_DAYS

PARAMS7 = list(PRIOR_BOUNDS)
PARAMS8 = PARAMS7 + ["k_delay"]

# Recharge-delay prior for the structural alternative.
DELAY_BOUNDS = ("loguniform", 0.005, 1.0)

# Features intentionally emphasize hydrograph dynamics / memory.
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

DOMAIN_FEATURES = DYN_FEATURES + [
    "p_mean",
    "p_p95",
    "wet_fraction",
    "p_cv",
    "t_mean",
    "t_sd",
    "pet_mean",
]


# ---------------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------------

def safe_id(x):
    return re.sub(r"[^A-Za-z0-9_.()-]+", "_", str(x))


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def civil_date(series):
    return pd.to_datetime(
        series.astype(str).str.slice(0, 10),
        format="%Y-%m-%d",
        errors="coerce",
    )


def boolish(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def autocorr(x, lag):
    s = pd.Series(np.asarray(x, float))
    return float(s.autocorr(lag)) if s.notna().sum() > lag + 5 else np.nan


def max_crosscorr_p_dlogq(p, q, maxlag=14):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    lq = np.log1p(np.maximum(q, 0))
    dq = np.r_[np.nan, np.diff(lq)]
    vals = []
    for lag in range(maxlag + 1):
        vals.append(
            pd.Series(dq).corr(pd.Series(p).shift(lag))
        )
    vals = np.asarray(vals, float)
    return float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan


def descriptors(p, t, pet, q_raw, dates):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    pet = np.asarray(pet, float)
    q_raw = np.asarray(q_raw, float)
    dates = pd.DatetimeIndex(dates)

    q = (
        pd.Series(q_raw, index=dates)
        .interpolate(method="time", limit_direction="both")
        .to_numpy(float)
    )

    good = (
        np.isfinite(p)
        & np.isfinite(t)
        & np.isfinite(pet)
        & np.isfinite(q)
    )
    p, t, pet, q, dates = (
        p[good], t[good], pet[good], q[good], dates[good]
    )

    if len(q) < 180:
        return None

    q = np.maximum(q, 0)
    logq = np.log1p(q)
    meanq = np.mean(q)

    denom = np.sum(q)
    flash = (
        float(np.sum(np.abs(np.diff(q))) / denom)
        if denom > 0 else np.nan
    )

    pmean = np.mean(p)
    p_cv = (
        float(np.std(p, ddof=1) / pmean)
        if pmean > 0 else np.nan
    )

    month = (
        pd.DataFrame({"logq": logq, "month": dates.month})
        .groupby("month")["logq"]
        .mean()
    )
    seasonal_amp = (
        float(month.max() - month.min())
        if len(month) >= 6 else np.nan
    )

    return {
        "logq_median": float(np.median(logq)),
        "logq_iqr": float(np.quantile(logq, .75) - np.quantile(logq, .25)),
        "q_cv": float(np.std(q, ddof=1) / meanq) if meanq > 0 else np.nan,
        "flashiness": flash,
        "q_lag1": autocorr(q, 1),
        "q_lag7": autocorr(q, 7),
        "q_lag30": autocorr(q, 30),
        "zero_fraction": float((q == 0).mean()),
        "corr_p_dlogq": float(
            pd.Series(np.r_[np.nan, np.diff(logq)]).corr(pd.Series(p))
        ),
        "max_corr_p_dlogq_lag0_14": max_crosscorr_p_dlogq(p, q, 14),
        "seasonal_logq_amplitude": seasonal_amp,
        "p_mean": float(np.mean(p)),
        "p_p95": float(np.quantile(p, .95)),
        "wet_fraction": float((p > .1).mean()),
        "p_cv": p_cv,
        "t_mean": float(np.mean(t)),
        "t_sd": float(np.std(t, ddof=1)),
        "pet_mean": float(np.mean(pet)),
    }


# ---------------------------------------------------------------------------
# REAL DATA / FORCING
# ---------------------------------------------------------------------------

def load_forcing(path):
    d = pd.read_csv(path)
    if "date" not in d.columns:
        raise KeyError(f"{path}: manca date")

    d["date"] = civil_date(d["date"])
    d = d.dropna(subset=["date"]).sort_values("date").set_index("date")

    if "precip_clean_mm" not in d.columns:
        d["precip_clean_mm"] = pd.to_numeric(
            d["precip_mm"], errors="coerce"
        ).clip(lower=0)

    return d


def forcing_complete(d):
    good = (
        pd.to_numeric(d["tmean_c"], errors="coerce").notna()
        & pd.to_numeric(d["precip_mm"], errors="coerce").notna()
        & pd.to_numeric(d["pet_oudin_mm"], errors="coerce").notna()
    )

    if "temperature_day_complete" in d.columns:
        good &= boolish(d["temperature_day_complete"])

    if "precip_day_complete" in d.columns:
        good &= boolish(d["precip_day_complete"])

    return good


def contiguous_valid_starts(d, length):
    good = forcing_complete(d).to_numpy(bool)
    if len(good) < length:
        return np.array([], dtype=int)

    c = np.concatenate([[0], np.cumsum(good.astype(int))])
    counts = c[length:] - c[:-length]
    return np.flatnonzero(counts == length)


def build_forcing_map(cohort, forcing_dir):
    fmap = {}
    for wid in cohort["wokas_id"].astype(str):
        p = Path(forcing_dir) / f"{safe_id(wid)}.csv"
        if p.exists():
            fmap[wid] = load_forcing(p)
    return fmap


def build_real_descriptors(
    cohort,
    discharge_dir,
    forcing_dir,
    context_days,
    windows_per_site=5,
    min_q_complete=.95,
):
    rows = []

    for wid in cohort["wokas_id"].astype(str):
        qp = Path(discharge_dir) / f"{safe_id(wid)}.csv"
        fp = Path(forcing_dir) / f"{safe_id(wid)}.csv"

        if not qp.exists() or not fp.exists():
            continue

        qd = pd.read_csv(qp)
        fd = pd.read_csv(fp)

        qd["date"] = civil_date(qd["date"])
        fd["date"] = civil_date(fd["date"])

        qcol = (
            "q_clean_m3s"
            if "q_clean_m3s" in qd.columns
            else "q_m3s"
        )

        usef = ["date", "precip_mm", "tmean_c", "pet_oudin_mm"]
        d = (
            qd[["date", qcol]]
            .dropna(subset=["date"])
            .merge(
                fd[usef].dropna(subset=["date"]),
                on="date",
                how="inner",
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(d) < context_days:
            continue

        candidates = []
        step = max(1, context_days // 4)

        for st in range(0, len(d) - context_days + 1, step):
            sub = d.iloc[st:st+context_days]
            qcomp = pd.to_numeric(
                sub[qcol], errors="coerce"
            ).notna().mean()

            forcing_ok = (
                pd.to_numeric(sub["precip_mm"], errors="coerce").notna().all()
                and pd.to_numeric(sub["tmean_c"], errors="coerce").notna().all()
                and pd.to_numeric(sub["pet_oudin_mm"], errors="coerce").notna().all()
            )

            if qcomp >= min_q_complete and forcing_ok:
                candidates.append(st)

        if not candidates:
            continue

        pick = np.linspace(
            0, len(candidates)-1,
            min(windows_per_site, len(candidates))
        ).round().astype(int)

        for wi, ci in enumerate(sorted(set(pick.tolist()))):
            st = candidates[ci]
            sub = d.iloc[st:st+context_days]

            desc = descriptors(
                pd.to_numeric(sub["precip_mm"], errors="coerce"),
                pd.to_numeric(sub["tmean_c"], errors="coerce"),
                pd.to_numeric(sub["pet_oudin_mm"], errors="coerce"),
                pd.to_numeric(sub[qcol], errors="coerce"),
                sub["date"],
            )

            if desc is None:
                continue

            rows.append({
                "domain": 1,
                "site_id": wid,
                "country": wid.split("-")[0],
                "window": wi,
                **desc,
            })

    return pd.DataFrame(rows)


def country_stratified_bridge_split(site_ids, fraction, seed):
    rng = np.random.default_rng(seed)
    tab = pd.DataFrame({"site_id": sorted(set(map(str, site_ids)))})
    tab["country"] = tab["site_id"].str.split("-").str[0]

    bridge, locked = [], []

    for country, g in tab.groupby("country"):
        arr = g["site_id"].to_numpy().copy()
        rng.shuffle(arr)

        n_bridge = max(1, int(round(len(arr) * fraction)))
        n_bridge = min(n_bridge, len(arr)-1) if len(arr) > 1 else 1

        bridge.extend(arr[:n_bridge].tolist())
        locked.extend(arr[n_bridge:].tolist())

    return sorted(bridge), sorted(locked)


# ---------------------------------------------------------------------------
# PRIOR QUANTILES / CEM
# ---------------------------------------------------------------------------

def q_to_value(name, u, model_name):
    if name == "k_delay":
        fam, lo, hi = DELAY_BOUNDS
    else:
        fam, lo, hi = PRIOR_BOUNDS[name]

    u = np.clip(float(u), 1e-6, 1-1e-6)

    if fam == "uniform":
        return lo + u * (hi-lo)

    if fam == "loguniform":
        return math.exp(
            math.log(lo) + u * (math.log(hi)-math.log(lo))
        )

    raise ValueError(fam)


def u_to_params(u, model_name):
    names = PARAMS7 if model_name == "M7" else PARAMS8
    vals = {name: q_to_value(name, u[i], model_name) for i, name in enumerate(names)}

    p = KarstParams(**{k: vals[k] for k in PARAMS7})

    if p.k_fast <= p.k_slow:
        return None, vals

    return p, vals


def logit_u(u):
    u = np.clip(np.asarray(u, float), 1e-5, 1-1e-5)
    return np.log(u/(1-u))


def sigmoid_z(z):
    z = np.clip(np.asarray(z, float), -12, 12)
    return 1/(1+np.exp(-z))


def sample_u_from_proposal(rng, n, dim, mean=None, cov=None):
    if mean is None or cov is None:
        # Exact uniform initial search in prior-quantile space.
        return rng.uniform(.01, .99, size=(n, dim))

    z = rng.multivariate_normal(mean, cov, size=n)

    # Exploration mixture: 15% broad uniform every round.
    m = max(1, int(round(.15*n)))
    z[:m] = logit_u(
        rng.uniform(.01, .99, size=(m, dim))
    )

    return np.clip(sigmoid_z(z), .005, .995)


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------

def simulate_m8(forcing, p, k_delay):
    """
    M7 + one delayed-recharge reservoir.

    Recharge generated by rainfall/snow/PET is accumulated in D and released
    as k_delay * D per day. Fast/slow partitioning acts on released recharge.
    """
    idx = pd.DatetimeIndex(forcing.index)
    n = len(forcing)

    P = pd.to_numeric(
        forcing["precip_clean_mm"], errors="coerce"
    ).to_numpy(float)
    T = pd.to_numeric(
        forcing["tmean_c"], errors="coerce"
    ).to_numpy(float)
    PET = pd.to_numeric(
        forcing["pet_oudin_mm"], errors="coerce"
    ).to_numpy(float)

    Sslow = 0.0
    Sfast = 0.0
    SWE = 0.0
    D = 0.0

    q = np.full(n, np.nan)

    for i in range(n):
        if not (
            np.isfinite(P[i])
            and np.isfinite(T[i])
            and np.isfinite(PET[i])
        ):
            continue

        snow_frac = sigmoid(
            (T_SNOW_C - T[i]) / SNOW_TRANSITION_C
        )
        snowfall = max(0.0, P[i]) * snow_frac
        rainfall = max(0.0, P[i]) - snowfall

        SWE += snowfall
        potential_melt = (
            DEGREE_DAY_FACTOR_MM_C_DAY
            * max(T[i] - T_MELT_C, 0.0)
        )
        melt = min(SWE, potential_melt)
        SWE -= melt

        liquid = rainfall + melt
        recharge_raw = max(
            0.0,
            liquid - p.lambda_et * max(PET[i], 0.0)
        )

        D += recharge_raw
        released = min(D, k_delay * D)
        D -= released

        fast_fraction = sigmoid(
            (released - p.tau_fast_mm)
            / FAST_GATE_WIDTH_MM
        )

        I_fast = fast_fraction * released
        I_slow = released - I_fast

        Sslow += I_slow
        Sfast += I_fast

        qslow = min(
            Sslow,
            p.k_slow * max(Sslow, 0.0)**p.alpha_slow
        )
        qfast = min(
            Sfast,
            p.k_fast * max(Sfast, 0.0)**p.alpha_fast
        )

        Sslow -= qslow
        Sfast -= qfast

        q[i] = (
            (qslow + qfast)
            * p.A_eff_km2
            * M3_PER_MM_KM2
            / SECONDS_PER_DAY
        )

    return pd.Series(q, index=idx)


def run_model(forcing, u, model_name):
    p, vals = u_to_params(u, model_name)

    if p is None:
        return None, vals

    if model_name == "M7":
        q = simulate_m7(
            forcing, p
        )["q_phys_m3s"]
    elif model_name == "M8":
        q = simulate_m8(
            forcing, p, vals["k_delay"]
        )
    else:
        raise ValueError(model_name)

    return q.to_numpy(float), vals


# ---------------------------------------------------------------------------
# SIMULATION / OBJECTIVE
# ---------------------------------------------------------------------------

def choose_forcing_window(rng, forcing_map, allowed_sites, total_days):
    eligible = []

    for wid in allowed_sites:
        d = forcing_map.get(wid)
        if d is None:
            continue

        starts = contiguous_valid_starts(d, total_days)

        if len(starts):
            eligible.append((wid, starts))

    if not eligible:
        raise RuntimeError("No valid forcing windows in requested site set.")

    wid, starts = eligible[int(rng.integers(0, len(eligible)))]
    st = int(starts[int(rng.integers(0, len(starts)))])

    return wid, d if False else forcing_map[wid].iloc[st:st+total_days].copy()


def robust_reference(real_bridge):
    med = real_bridge[DYN_FEATURES].median()
    iqr = (
        real_bridge[DYN_FEATURES].quantile(.75)
        - real_bridge[DYN_FEATURES].quantile(.25)
    )
    iqr[iqr < 1e-6] = 1.0

    Z = (
        (real_bridge[DYN_FEATURES] - med) / iqr
    ).replace([np.inf,-np.inf], np.nan).fillna(0)

    return med, iqr, Z


def descriptor_distance(desc, med, iqr, nn):
    x = pd.DataFrame([desc])[DYN_FEATURES]
    z = (
        (x - med) / iqr
    ).replace([np.inf,-np.inf], np.nan).fillna(0).to_numpy(float)

    dist, _ = nn.kneighbors(z)
    return float(dist.mean())


def evaluate_candidate(
    rng,
    u,
    model_name,
    forcing_map,
    allowed_sites,
    med,
    iqr,
    nn,
):
    try:
        wid, forcing = choose_forcing_window(
            rng, forcing_map, allowed_sites, TOTAL_DAYS
        )

        q_all, vals = run_model(forcing, u, model_name)

        if q_all is None:
            return None

        sl = slice(WARMUP_DAYS, TOTAL_DAYS)

        desc = descriptors(
            forcing["precip_mm"].to_numpy(float)[sl],
            forcing["tmean_c"].to_numpy(float)[sl],
            forcing["pet_oudin_mm"].to_numpy(float)[sl],
            q_all[sl],
            forcing.index[sl],
        )

        if desc is None:
            return None

        score = descriptor_distance(
            desc, med, iqr, nn
        )

        return {
            "site_id": wid,
            "score": score,
            **{f"u_{name}": float(u[i]) for i, name in enumerate(
                PARAMS7 if model_name == "M7" else PARAMS8
            )},
            **desc,
        }

    except Exception:
        return None


def cem_fit(
    model_name,
    forcing_map,
    bridge_sites,
    real_bridge,
    candidates,
    rounds,
    elite_fraction,
    seed,
    out_dir,
):
    rng = np.random.default_rng(seed)

    dim = 7 if model_name == "M7" else 8
    mean = None
    cov = None

    med, iqr, Zref = robust_reference(real_bridge)
    nn = NearestNeighbors(
        n_neighbors=min(7, len(Zref))
    ).fit(Zref.to_numpy(float))

    history = []
    final_elites = None

    for rnd in range(1, rounds+1):
        print(f"\n{model_name} CEM round {rnd}/{rounds}")

        U = sample_u_from_proposal(
            rng, candidates, dim, mean, cov
        )

        rows = []

        for i, u in enumerate(U):
            rec = evaluate_candidate(
                rng, u, model_name,
                forcing_map, bridge_sites,
                med, iqr, nn
            )
            if rec is not None:
                rec["round"] = rnd
                rows.append(rec)

            if (i+1) % 100 == 0:
                print(f"  {i+1}/{candidates}")

        tab = pd.DataFrame(rows)

        if len(tab) < max(30, int(candidates*.2)):
            raise RuntimeError(
                f"{model_name}: too few valid CEM simulations ({len(tab)})"
            )

        n_elite = max(20, int(round(len(tab)*elite_fraction)))
        elite = tab.nsmallest(n_elite, "score").copy()

        names = PARAMS7 if model_name == "M7" else PARAMS8
        Uelite = elite[[f"u_{k}" for k in names]].to_numpy(float)
        Z = logit_u(Uelite)

        new_mean = Z.mean(axis=0)
        new_cov = np.cov(Z, rowvar=False)

        # Stabilize covariance.
        new_cov = np.atleast_2d(new_cov)
        new_cov += np.eye(dim)*0.05

        if mean is None:
            mean = new_mean
            cov = new_cov
        else:
            # Conservative CEM update; avoid premature collapse.
            mean = .65*mean + .35*new_mean
            cov = .65*cov + .35*new_cov

        history.append({
            "model": model_name,
            "round": rnd,
            "n_valid": int(len(tab)),
            "n_elite": int(len(elite)),
            "score_median": float(tab["score"].median()),
            "score_elite_median": float(elite["score"].median()),
            "score_elite_p90": float(elite["score"].quantile(.90)),
        })

        tab.to_csv(
            out_dir / f"{model_name}_round_{rnd}_candidates.csv",
            index=False
        )
        elite.to_csv(
            out_dir / f"{model_name}_round_{rnd}_elites.csv",
            index=False
        )

        final_elites = elite

        print(
            "  median score:",
            round(tab["score"].median(), 3),
            "elite:",
            round(elite["score"].median(), 3)
        )

    proposal = {
        "model": model_name,
        "parameter_names": PARAMS7 if model_name == "M7" else PARAMS8,
        "logit_u_mean": mean.tolist(),
        "logit_u_cov": cov.tolist(),
        "elite_fraction": elite_fraction,
        "bridge_sites": sorted(map(str, bridge_sites)),
    }

    return proposal, pd.DataFrame(history), final_elites


def sample_from_learned_proposal(rng, proposal, n):
    mean = np.asarray(proposal["logit_u_mean"], float)
    cov = np.asarray(proposal["logit_u_cov"], float)

    z = rng.multivariate_normal(mean, cov, size=n)
    u = np.clip(sigmoid_z(z), .005, .995)

    return u


def generate_locked_sims(
    model_name,
    proposal,
    forcing_map,
    locked_sites,
    n,
    seed,
):
    rng = np.random.default_rng(seed)
    U = sample_from_learned_proposal(rng, proposal, n)

    rows = []

    for i, u in enumerate(U):
        try:
            wid, forcing = choose_forcing_window(
                rng, forcing_map, locked_sites, TOTAL_DAYS
            )

            q_all, vals = run_model(
                forcing, u, model_name
            )

            if q_all is None:
                continue

            sl = slice(WARMUP_DAYS, TOTAL_DAYS)

            desc = descriptors(
                forcing["precip_mm"].to_numpy(float)[sl],
                forcing["tmean_c"].to_numpy(float)[sl],
                forcing["pet_oudin_mm"].to_numpy(float)[sl],
                q_all[sl],
                forcing.index[sl],
            )

            if desc is None:
                continue

            rows.append({
                "domain": 0,
                "site_id": wid,
                "simulation": i,
                **desc,
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LOCKED DOMAIN GAP
# ---------------------------------------------------------------------------

def rbf_mmd2(X, Y):
    Z = np.vstack([X, Y])
    rng = np.random.default_rng(1234)

    take = rng.choice(
        len(Z), min(len(Z), 600), replace=False
    )
    A = Z[take]

    d2 = np.sum(
        (A[:,None,:] - A[None,:,:])**2,
        axis=2,
    )

    pos = d2[d2 > 0]
    med = np.median(pos) if len(pos) else 1.0
    gamma = 1.0 / max(2*med, 1e-12)

    def ker(A, B):
        dd = np.sum(
            (A[:,None,:] - B[None,:,:])**2,
            axis=2,
        )
        return np.exp(-gamma*dd)

    Kxx = ker(X, X)
    Kyy = ker(Y, Y)
    Kxy = ker(X, Y)

    nx, ny = len(X), len(Y)

    return float(
        (Kxx.sum()-np.trace(Kxx))/max(nx*(nx-1),1)
        + (Kyy.sum()-np.trace(Kyy))/max(ny*(ny-1),1)
        - 2*Kxy.mean()
    )


def locked_domain_metrics(syn, real):
    real = real.copy()
    real["domain"] = 1

    alltab = pd.concat(
        [syn, real],
        ignore_index=True,
        sort=False,
    )

    X = (
        alltab[DOMAIN_FEATURES]
        .replace([np.inf,-np.inf], np.nan)
    )
    X = X.fillna(X.median(numeric_only=True))

    y = alltab["domain"].astype(int).to_numpy()
    groups = alltab["site_id"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=5)
    probs = np.full(len(alltab), np.nan)
    folds = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        if len(np.unique(y[te])) < 2:
            continue

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000)
        )
        model.fit(X.iloc[tr], y[tr])

        pr = model.predict_proba(X.iloc[te])[:,1]
        probs[te] = pr

        folds.append({
            "fold": fold,
            "auc": float(roc_auc_score(y[te], pr)),
            "n": int(len(te)),
            "sites": int(len(np.unique(groups[te]))),
        })

    good = np.isfinite(probs)
    pooled_auc = float(
        roc_auc_score(y[good], probs[good])
    )

    med = X.median()
    iqr = X.quantile(.75)-X.quantile(.25)
    iqr[iqr < 1e-6] = 1.0

    Z = (X-med)/iqr
    Zs = Z[alltab["domain"]==0].to_numpy(float)
    Zr = Z[alltab["domain"]==1].to_numpy(float)

    mmd = rbf_mmd2(Zs, Zr)

    shifts = []
    for f in DOMAIN_FEATURES:
        a = Z.loc[alltab["domain"]==0, f]
        b = Z.loc[alltab["domain"]==1, f]

        shifts.append({
            "feature": f,
            "synthetic_median": float(
                syn[f].median()
            ),
            "real_median": float(
                real[f].median()
            ),
            "abs_median_shift_robust_z": float(
                abs(a.median()-b.median())
            ),
        })

    shifts = pd.DataFrame(shifts).sort_values(
        "abs_median_shift_robust_z",
        ascending=False
    )

    return pooled_auc, mmd, pd.DataFrame(folds), shifts


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--cohort",
        default="karst_inverse_study/00_inventory/cohort_core_20y95.csv"
    )
    ap.add_argument(
        "--discharge-dir",
        default="karst_inverse_study/01_discharge"
    )
    ap.add_argument(
        "--forcing-dir",
        default="karst_inverse_study/02_forcing_arco/daily"
    )
    ap.add_argument(
        "--out",
        default="karst_inverse_study/12_domain_bridge"
    )

    ap.add_argument("--bridge-fraction", type=float, default=1/3)
    ap.add_argument("--real-windows", type=int, default=5)
    ap.add_argument("--min-q-complete", type=float, default=.95)

    ap.add_argument("--candidates", type=int, default=700)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--elite-fraction", type=float, default=.15)
    ap.add_argument("--locked-sims", type=int, default=700)

    ap.add_argument("--seed", type=int, default=20260812)

    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort)
    forcing_map = build_forcing_map(
        cohort, args.forcing_dir
    )

    print("\n=== REAL DESCRIPTORS ===")
    real = build_real_descriptors(
        cohort,
        args.discharge_dir,
        args.forcing_dir,
        CONTEXT_DAYS,
        windows_per_site=args.real_windows,
        min_q_complete=args.min_q_complete,
    )

    if real["site_id"].nunique() < 20:
        raise RuntimeError(
            f"Too few real sites with descriptors: {real['site_id'].nunique()}"
        )

    bridge_sites, locked_sites = country_stratified_bridge_split(
        real["site_id"].unique(),
        args.bridge_fraction,
        args.seed,
    )

    bridge = real[
        real["site_id"].isin(bridge_sites)
    ].copy()

    locked = real[
        real["site_id"].isin(locked_sites)
    ].copy()

    bridge.to_csv(root / "real_bridge_descriptors.csv", index=False)
    locked.to_csv(root / "real_locked_descriptors.csv", index=False)

    save_json({
        "seed": args.seed,
        "bridge_fraction": args.bridge_fraction,
        "bridge_sites": bridge_sites,
        "locked_sites": locked_sites,
        "n_bridge_sites": len(bridge_sites),
        "n_locked_sites": len(locked_sites),
        "country_counts_bridge": (
            pd.Series(bridge_sites).str.split("-").str[0]
            .value_counts().to_dict()
        ),
        "country_counts_locked": (
            pd.Series(locked_sites).str.split("-").str[0]
            .value_counts().to_dict()
        ),
    }, root / "bridge_locked_split.json")

    print("bridge sites:", len(bridge_sites))
    print("locked sites:", len(locked_sites))

    all_results = {}

    for mi, model_name in enumerate(["M7", "M8"]):
        print("\n" + "="*80)
        print(f"{model_name} — CEM POPULATION BRIDGE")
        print("="*80)

        mdir = root / model_name
        mdir.mkdir(parents=True, exist_ok=True)

        proposal, hist, elites = cem_fit(
            model_name,
            forcing_map,
            bridge_sites,
            bridge,
            candidates=args.candidates,
            rounds=args.rounds,
            elite_fraction=args.elite_fraction,
            seed=args.seed + 1000*mi,
            out_dir=mdir,
        )

        save_json(
            proposal,
            mdir / "learned_population_prior.json"
        )
        hist.to_csv(
            mdir / "cem_history.csv",
            index=False
        )
        elites.to_csv(
            mdir / "final_elites.csv",
            index=False
        )

        print(f"\n{model_name} — LOCKED EVALUATION")

        syn_locked = generate_locked_sims(
            model_name,
            proposal,
            forcing_map,
            locked_sites,
            args.locked_sims,
            seed=args.seed + 5000 + 1000*mi,
        )

        syn_locked.to_csv(
            mdir / "locked_synthetic_descriptors.csv",
            index=False
        )

        auc, mmd, folds, shifts = locked_domain_metrics(
            syn_locked,
            locked,
        )

        folds.to_csv(
            mdir / "locked_domain_folds.csv",
            index=False
        )
        shifts.to_csv(
            mdir / "locked_feature_shifts.csv",
            index=False
        )

        report = {
            "model": model_name,
            "n_bridge_sites": len(bridge_sites),
            "n_locked_sites": len(locked_sites),
            "n_locked_synthetic_windows": int(len(syn_locked)),
            "n_locked_real_windows": int(len(locked)),
            "locked_grouped_domain_auc": float(auc),
            "locked_rbf_mmd2": float(mmd),
            "pass_auc_le_0_75": bool(auc <= .75),
            "top_5_locked_shifts": shifts.head(5).to_dict("records"),
        }

        save_json(
            report,
            mdir / "locked_domain_gate.json"
        )

        all_results[model_name] = report

        print("locked grouped AUC:", round(auc, 3))
        print("locked MMD^2:", round(mmd, 4))
        print("PASS:", report["pass_auc_le_0_75"])
        print("\nLargest locked shifts:")
        print(shifts.head(8).to_string(index=False))

    m7_pass = all_results["M7"]["pass_auc_le_0_75"]
    m8_pass = all_results["M8"]["pass_auc_le_0_75"]

    if m7_pass:
        decision = {
            "decision": "KEEP_M7",
            "message": (
                "The current 7-parameter minimal physics can bridge the real "
                "domain after learning a population prior on BRIDGE sites. "
                "Do not add recharge-delay physics."
            ),
        }
    elif m8_pass:
        decision = {
            "decision": "UPGRADE_TO_M8_RECHARGE_DELAY",
            "message": (
                "The 7-parameter model fails on LOCKED sites but the one-delay "
                "extension passes. Explicit recharge memory is structurally "
                "supported before returning to APIC."
            ),
        }
    else:
        decision = {
            "decision": "DEEPER_PHYSICS_REVISION_REQUIRED",
            "message": (
                "Neither population-prior adaptation nor a single recharge-delay "
                "store closes the LOCKED domain gap. Do not run APIC multi-seed "
                "or infer real aquifer parameters yet."
            ),
        }

    summary = {
        "M7": all_results["M7"],
        "M8": all_results["M8"],
        **decision,
    }

    save_json(
        summary,
        root / "DOMAIN_BRIDGE_DECISION.json"
    )

    print("\n" + "="*80)
    print("DOMAIN BRIDGE — FINAL DECISION")
    print("="*80)
    print("M7 locked AUC:", round(all_results["M7"]["locked_grouped_domain_auc"], 3))
    print("M8 locked AUC:", round(all_results["M8"]["locked_grouped_domain_auc"], 3))
    print("DECISION:", decision["decision"])
    print(decision["message"])


if __name__ == "__main__":
    main()
