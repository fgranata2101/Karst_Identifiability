#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
karst_matrix_conduit_exchange_test.py
=====================================

NEXT STRUCTURAL TEST AFTER M7/M8/M9.

Scientific question
-------------------
The previous bridge/locked tests showed:

- M7: too flashy / too little memory.
- M8: whole-recharge delay over-smoothed the hydrograph.
- M9: partial epikarst delay improved the domain gap, but remained too smooth,
      too weakly seasonal, and too directly coupled to rainfall.

This script therefore tests whether the missing ingredient is TOPOLOGY:
the slow domain should not discharge independently to the spring, but should
feed the conduit domain.

Two parsimonious alternatives are compared on the SAME BRIDGE/LOCKED split.

M10 — MATRIX -> CONDUIT CASCADE
-------------------------------
Same 7 parameters as the original model.

Recharge partition:
    I_c = phi(R) R
    I_m = (1-phi(R)) R

Matrix drainage:
    X_mc = k_slow * S_m^alpha_slow
moves into the conduit storage.

Spring discharge:
    Q = k_fast * S_c^alpha_fast

Thus the spring is the outlet of the conduit domain only.

M11 — BIDIRECTIONAL MATRIX <-> CONDUIT EXCHANGE
------------------------------------------------
Adds only one parameter rho_mc, a storage/head scaling ratio.

Recharge enters matrix/conduit exactly as in M10.

Exchange uses a storage-gradient proxy:
    Delta = S_m - rho_mc * S_c

    X_mc = sign(Delta) * k_slow * |Delta|^alpha_slow

Positive X_mc:
    matrix -> conduit
Negative X_mc:
    conduit -> matrix

Spring discharge remains conduit outflow only.

The bidirectional option is motivated as a conceptual analogue of transient
matrix-conduit exchange; it is NOT claimed to be a literal hydraulic-head model.

Bridge/locked protocol
----------------------
- Reuses the EXACT bridge/locked real split already created in
  karst_inverse_study/12_domain_bridge/.
- Learns a population proposal ONLY on BRIDGE descriptors using CEM.
- Evaluates synthetic vs real ONLY on LOCKED sites.
- Uses site-grouped domain-classifier AUC and MMD.

Decision rule
-------------
M10 AUC <= 0.75:
    KEEP_M10_CASCADE

M10 > 0.75 and M11 <= 0.75:
    KEEP_M11_BIDIRECTIONAL_EXCHANGE

Both > 0.75:
    MORE_EXPLICIT_STRUCTURE_REQUIRED

Execution in Spyder
-------------------
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/karst_matrix_conduit_exchange_test.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"

Smoke test
----------
%runfile ".../karst_matrix_conduit_exchange_test.py" --args "--candidates 250 --rounds 2 --locked-sims 250" --wdir "..."

Requirements
------------
numpy pandas scipy scikit-learn
karst_minimal_physics.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import re
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
    sigmoid,
    T_SNOW_C,
    SNOW_TRANSITION_C,
    T_MELT_C,
    DEGREE_DAY_FACTOR_MM_C_DAY,
    FAST_GATE_WIDTH_MM,
    M3_PER_MM_KM2,
    SECONDS_PER_DAY,
)

# -----------------------------------------------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------------------------------------------

CONTEXT_DAYS = 365
WARMUP_DAYS = 10 * 365 + 3
TOTAL_DAYS = WARMUP_DAYS + CONTEXT_DAYS

PARAMS7 = list(PRIOR_BOUNDS)
PARAMS11 = PARAMS7 + ["rho_mc"]

# Converts conduit storage to matrix-equivalent storage for the exchange proxy.
# Broad design prior; this is deliberately not outcome-tuned.
RHO_BOUNDS = ("loguniform", 0.05, 20.0)

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


# -----------------------------------------------------------------------------
# GENERIC HELPERS
# -----------------------------------------------------------------------------

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

    p = p[good]
    t = t[good]
    pet = pet[good]
    q = q[good]
    dates = dates[good]

    if len(q) < 180:
        return None

    q = np.maximum(q, 0.0)
    logq = np.log1p(q)

    meanq = float(np.mean(q))
    denom = float(np.sum(q))

    flashiness = (
        float(np.sum(np.abs(np.diff(q))) / denom)
        if denom > 0 else np.nan
    )

    pmean = float(np.mean(p))
    pcv = (
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
        if len(month) >= 6
        else np.nan
    )

    return {
        "logq_median": float(np.median(logq)),
        "logq_iqr": float(
            np.quantile(logq, .75) - np.quantile(logq, .25)
        ),
        "q_cv": (
            float(np.std(q, ddof=1) / meanq)
            if meanq > 0 else np.nan
        ),
        "flashiness": flashiness,
        "q_lag1": autocorr(q, 1),
        "q_lag7": autocorr(q, 7),
        "q_lag30": autocorr(q, 30),
        "zero_fraction": float((q == 0).mean()),
        "corr_p_dlogq": float(
            pd.Series(np.r_[np.nan, np.diff(logq)]).corr(pd.Series(p))
        ),
        "max_corr_p_dlogq_lag0_14": max_crosscorr_p_dlogq(p, q, 14),
        "seasonal_logq_amplitude": seasonal_amp,
        "p_mean": pmean,
        "p_p95": float(np.quantile(p, .95)),
        "wet_fraction": float((p > .1).mean()),
        "p_cv": pcv,
        "t_mean": float(np.mean(t)),
        "t_sd": float(np.std(t, ddof=1)),
        "pet_mean": float(np.mean(pet)),
    }


# -----------------------------------------------------------------------------
# PARAMETER TRANSFORMS
# -----------------------------------------------------------------------------

def q_to_value(name, u):
    if name == "rho_mc":
        fam, lo, hi = RHO_BOUNDS
    else:
        fam, lo, hi = PRIOR_BOUNDS[name]

    u = np.clip(float(u), 1e-6, 1 - 1e-6)

    if fam == "uniform":
        return lo + u * (hi - lo)

    if fam == "loguniform":
        return math.exp(
            math.log(lo)
            + u * (math.log(hi) - math.log(lo))
        )

    raise ValueError(fam)


def u_to_params(u, model_name):
    names = PARAMS7 if model_name == "M10" else PARAMS11

    vals = {
        name: q_to_value(name, u[i])
        for i, name in enumerate(names)
    }

    p = KarstParams(**{
        k: vals[k]
        for k in PARAMS7
    })

    if p.k_fast <= p.k_slow:
        return None, vals

    return p, vals


def logit_u(u):
    u = np.clip(np.asarray(u, float), 1e-5, 1 - 1e-5)
    return np.log(u / (1 - u))


def sigmoid_z(z):
    z = np.clip(np.asarray(z, float), -12, 12)
    return 1.0 / (1.0 + np.exp(-z))


def sample_u(rng, n, dim, mean=None, cov=None):
    if mean is None:
        return rng.uniform(.01, .99, size=(n, dim))

    z = rng.multivariate_normal(mean, cov, size=n)

    # Persistent broad exploration to avoid premature CEM collapse.
    m = max(1, int(round(.15 * n)))
    z[:m] = logit_u(
        rng.uniform(.01, .99, size=(m, dim))
    )

    return np.clip(
        sigmoid_z(z),
        .005,
        .995
    )


# -----------------------------------------------------------------------------
# FORCING
# -----------------------------------------------------------------------------

def load_forcing(path):
    d = pd.read_csv(path)

    if "date" not in d.columns:
        raise KeyError(f"{path}: missing date")

    d["date"] = civil_date(d["date"])

    d = (
        d.dropna(subset=["date"])
        .sort_values("date")
        .set_index("date")
    )

    if "precip_clean_mm" not in d.columns:
        d["precip_clean_mm"] = pd.to_numeric(
            d["precip_mm"],
            errors="coerce"
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

    cs = np.concatenate([[0], np.cumsum(good.astype(int))])
    counts = cs[length:] - cs[:-length]

    return np.flatnonzero(counts == length)


def build_forcing_map(cohort, forcing_dir):
    fmap = {}
    startmap = {}

    for wid in cohort["wokas_id"].astype(str):
        p = Path(forcing_dir) / f"{safe_id(wid)}.csv"

        if not p.exists():
            continue

        d = load_forcing(p)
        starts = contiguous_valid_starts(d, TOTAL_DAYS)

        if len(starts):
            fmap[wid] = d
            startmap[wid] = starts

    return fmap, startmap


def choose_forcing_window(rng, fmap, startmap, sites):
    eligible = [
        s for s in sites
        if s in fmap and len(startmap.get(s, []))
    ]

    if not eligible:
        raise RuntimeError("No eligible forcing windows.")

    wid = eligible[int(rng.integers(0, len(eligible)))]
    starts = startmap[wid]
    st = int(
        starts[int(rng.integers(0, len(starts)))]
    )

    return wid, fmap[wid].iloc[
        st:st+TOTAL_DAYS
    ].copy()


# -----------------------------------------------------------------------------
# COMMON SNOW / EFFECTIVE RECHARGE
# -----------------------------------------------------------------------------

def effective_recharge_step(P, T, PET, SWE, p):
    snow_frac = sigmoid(
        (T_SNOW_C - T) / SNOW_TRANSITION_C
    )

    snowfall = max(0.0, P) * snow_frac
    rainfall = max(0.0, P) - snowfall

    SWE += snowfall

    potential_melt = (
        DEGREE_DAY_FACTOR_MM_C_DAY
        * max(T - T_MELT_C, 0.0)
    )

    melt = min(SWE, potential_melt)
    SWE -= melt

    liquid = rainfall + melt

    R = max(
        0.0,
        liquid - p.lambda_et * max(PET, 0.0)
    )

    phi = sigmoid(
        (R - p.tau_fast_mm)
        / FAST_GATE_WIDTH_MM
    )

    return R, phi, SWE


# -----------------------------------------------------------------------------
# M10 — UNIDIRECTIONAL MATRIX -> CONDUIT CASCADE
# -----------------------------------------------------------------------------

def simulate_m10(forcing, p):
    idx = pd.DatetimeIndex(forcing.index)
    n = len(forcing)

    P = pd.to_numeric(
        forcing["precip_clean_mm"],
        errors="coerce"
    ).to_numpy(float)

    T = pd.to_numeric(
        forcing["tmean_c"],
        errors="coerce"
    ).to_numpy(float)

    PET = pd.to_numeric(
        forcing["pet_oudin_mm"],
        errors="coerce"
    ).to_numpy(float)

    Sm = 0.0
    Sc = 0.0
    SWE = 0.0

    q = np.full(n, np.nan)

    for i in range(n):
        if not (
            np.isfinite(P[i])
            and np.isfinite(T[i])
            and np.isfinite(PET[i])
        ):
            continue

        R, phi, SWE = effective_recharge_step(
            P[i], T[i], PET[i], SWE, p
        )

        Ic = phi * R
        Im = (1.0 - phi) * R

        Sm += Im
        Sc += Ic

        # Matrix drainage feeds conduit; it is NOT direct spring discharge.
        Xmc = min(
            Sm,
            p.k_slow * max(Sm, 0.0)**p.alpha_slow
        )

        Sm -= Xmc
        Sc += Xmc

        qspring_mm = min(
            Sc,
            p.k_fast * max(Sc, 0.0)**p.alpha_fast
        )

        Sc -= qspring_mm

        q[i] = (
            qspring_mm
            * p.A_eff_km2
            * M3_PER_MM_KM2
            / SECONDS_PER_DAY
        )

    return q


# -----------------------------------------------------------------------------
# M11 — BIDIRECTIONAL MATRIX <-> CONDUIT EXCHANGE
# -----------------------------------------------------------------------------

def simulate_m11(forcing, p, rho_mc):
    idx = pd.DatetimeIndex(forcing.index)
    n = len(forcing)

    P = pd.to_numeric(
        forcing["precip_clean_mm"],
        errors="coerce"
    ).to_numpy(float)

    T = pd.to_numeric(
        forcing["tmean_c"],
        errors="coerce"
    ).to_numpy(float)

    PET = pd.to_numeric(
        forcing["pet_oudin_mm"],
        errors="coerce"
    ).to_numpy(float)

    Sm = 0.0
    Sc = 0.0
    SWE = 0.0

    q = np.full(n, np.nan)

    for i in range(n):
        if not (
            np.isfinite(P[i])
            and np.isfinite(T[i])
            and np.isfinite(PET[i])
        ):
            continue

        R, phi, SWE = effective_recharge_step(
            P[i], T[i], PET[i], SWE, p
        )

        Ic = phi * R
        Im = (1.0 - phi) * R

        Sm += Im
        Sc += Ic

        # Conceptual storage/head gradient.
        delta = Sm - rho_mc * Sc

        mag = (
            p.k_slow
            * abs(delta)**p.alpha_slow
        )

        if delta >= 0:
            # Matrix -> conduit
            X = min(Sm, mag)
            Sm -= X
            Sc += X
        else:
            # Conduit -> matrix
            X = min(Sc, mag)
            Sc -= X
            Sm += X

        qspring_mm = min(
            Sc,
            p.k_fast * max(Sc, 0.0)**p.alpha_fast
        )

        Sc -= qspring_mm

        q[i] = (
            qspring_mm
            * p.A_eff_km2
            * M3_PER_MM_KM2
            / SECONDS_PER_DAY
        )

    return q


def run_model(forcing, u, model_name):
    p, vals = u_to_params(
        u,
        model_name
    )

    if p is None:
        return None, vals

    if model_name == "M10":
        q = simulate_m10(
            forcing,
            p
        )

    elif model_name == "M11":
        q = simulate_m11(
            forcing,
            p,
            vals["rho_mc"]
        )

    else:
        raise ValueError(model_name)

    return q, vals


# -----------------------------------------------------------------------------
# BRIDGE OBJECTIVE
# -----------------------------------------------------------------------------

def robust_reference(real_bridge):
    med = real_bridge[DYN_FEATURES].median()

    iqr = (
        real_bridge[DYN_FEATURES].quantile(.75)
        - real_bridge[DYN_FEATURES].quantile(.25)
    )

    iqr[iqr < 1e-6] = 1.0

    Z = (
        (real_bridge[DYN_FEATURES] - med) / iqr
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return med, iqr, Z


def descriptor_distance(desc, med, iqr, nn):
    x = pd.DataFrame([desc])[DYN_FEATURES]

    z = (
        (x - med) / iqr
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0).to_numpy(float)

    dist, _ = nn.kneighbors(z)

    return float(
        dist.mean()
    )


def evaluate_candidate(
    rng,
    u,
    model_name,
    fmap,
    startmap,
    bridge_sites,
    med,
    iqr,
    nn,
):
    try:
        wid, forcing = choose_forcing_window(
            rng,
            fmap,
            startmap,
            bridge_sites
        )

        q, vals = run_model(
            forcing,
            u,
            model_name
        )

        if q is None:
            return None

        sl = slice(
            WARMUP_DAYS,
            TOTAL_DAYS
        )

        desc = descriptors(
            forcing["precip_mm"].to_numpy(float)[sl],
            forcing["tmean_c"].to_numpy(float)[sl],
            forcing["pet_oudin_mm"].to_numpy(float)[sl],
            q[sl],
            forcing.index[sl],
        )

        if desc is None:
            return None

        names = (
            PARAMS7
            if model_name == "M10"
            else PARAMS11
        )

        return {
            "site_id": wid,
            "score": descriptor_distance(
                desc,
                med,
                iqr,
                nn
            ),
            **{
                f"u_{name}": float(u[i])
                for i, name in enumerate(names)
            },
            **desc,
        }

    except Exception:
        return None


def cem_fit(
    model_name,
    bridge,
    bridge_sites,
    fmap,
    startmap,
    candidates,
    rounds,
    elite_fraction,
    seed,
    out,
):
    rng = np.random.default_rng(seed)

    names = (
        PARAMS7
        if model_name == "M10"
        else PARAMS11
    )
    dim = len(names)

    med, iqr, Zref = robust_reference(
        bridge
    )

    nn = NearestNeighbors(
        n_neighbors=min(
            7,
            len(Zref)
        )
    ).fit(
        Zref.to_numpy(float)
    )

    mean = None
    cov = None
    history = []

    for rnd in range(1, rounds + 1):
        print(
            f"\n{model_name} CEM round {rnd}/{rounds}"
        )

        U = sample_u(
            rng,
            candidates,
            dim,
            mean,
            cov
        )

        rows = []

        for i, u in enumerate(U):
            rec = evaluate_candidate(
                rng,
                u,
                model_name,
                fmap,
                startmap,
                bridge_sites,
                med,
                iqr,
                nn,
            )

            if rec is not None:
                rec["round"] = rnd
                rows.append(rec)

            if (i + 1) % 100 == 0:
                print(
                    f"  {i+1}/{candidates}"
                )

        tab = pd.DataFrame(rows)

        if len(tab) < max(
            30,
            int(.2 * candidates)
        ):
            raise RuntimeError(
                f"{model_name}: too few valid simulations ({len(tab)})"
            )

        n_elite = max(
            25,
            int(round(
                len(tab)
                * elite_fraction
            ))
        )

        elite = tab.nsmallest(
            n_elite,
            "score"
        ).copy()

        Ue = elite[
            [f"u_{k}" for k in names]
        ].to_numpy(float)

        Ze = logit_u(Ue)

        new_mean = Ze.mean(axis=0)
        new_cov = np.cov(
            Ze,
            rowvar=False
        )

        new_cov = (
            np.atleast_2d(new_cov)
            + np.eye(dim) * .05
        )

        if mean is None:
            mean = new_mean
            cov = new_cov
        else:
            mean = (
                .65 * mean
                + .35 * new_mean
            )
            cov = (
                .65 * cov
                + .35 * new_cov
            )

        tab.to_csv(
            out / f"{model_name}_round_{rnd}_candidates.csv",
            index=False
        )
        elite.to_csv(
            out / f"{model_name}_round_{rnd}_elites.csv",
            index=False
        )

        history.append({
            "round": rnd,
            "n_valid": int(len(tab)),
            "n_elite": int(len(elite)),
            "median_score": float(tab["score"].median()),
            "elite_median_score": float(elite["score"].median()),
        })

        print(
            "  median score:",
            round(tab["score"].median(), 3),
            "elite:",
            round(elite["score"].median(), 3)
        )

    proposal = {
        "model": model_name,
        "parameter_names": names,
        "logit_u_mean":
            mean.tolist(),
        "logit_u_cov":
            cov.tolist(),
        "elite_fraction":
            elite_fraction,
    }

    pd.DataFrame(
        history
    ).to_csv(
        out / f"{model_name}_cem_history.csv",
        index=False
    )

    save_json(
        proposal,
        out / f"{model_name}_population_prior.json"
    )

    return proposal


# -----------------------------------------------------------------------------
# LOCKED GENERATION
# -----------------------------------------------------------------------------

def sample_proposal(
    rng,
    proposal,
    n
):
    mean = np.asarray(
        proposal["logit_u_mean"],
        float
    )

    cov = np.asarray(
        proposal["logit_u_cov"],
        float
    )

    z = rng.multivariate_normal(
        mean,
        cov,
        size=n
    )

    return np.clip(
        sigmoid_z(z),
        .005,
        .995
    )


def generate_locked(
    model_name,
    proposal,
    fmap,
    startmap,
    locked_sites,
    n,
    seed,
):
    rng = np.random.default_rng(seed)

    U = sample_proposal(
        rng,
        proposal,
        n
    )

    rows = []

    for i, u in enumerate(U):
        try:
            wid, forcing = choose_forcing_window(
                rng,
                fmap,
                startmap,
                locked_sites
            )

            q, vals = run_model(
                forcing,
                u,
                model_name
            )

            if q is None:
                continue

            sl = slice(
                WARMUP_DAYS,
                TOTAL_DAYS
            )

            desc = descriptors(
                forcing["precip_mm"].to_numpy(float)[sl],
                forcing["tmean_c"].to_numpy(float)[sl],
                forcing["pet_oudin_mm"].to_numpy(float)[sl],
                q[sl],
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


# -----------------------------------------------------------------------------
# DOMAIN METRICS
# -----------------------------------------------------------------------------

def rbf_mmd2(X, Y):
    Z = np.vstack(
        [X, Y]
    )

    rng = np.random.default_rng(
        1234
    )

    take = rng.choice(
        len(Z),
        min(len(Z), 600),
        replace=False
    )

    A = Z[take]

    d2 = np.sum(
        (
            A[:, None, :]
            - A[None, :, :]
        ) ** 2,
        axis=2
    )

    pos = d2[d2 > 0]
    med = (
        np.median(pos)
        if len(pos)
        else 1.0
    )

    gamma = 1.0 / max(
        2 * med,
        1e-12
    )

    def ker(A, B):
        dd = np.sum(
            (
                A[:, None, :]
                - B[None, :, :]
            ) ** 2,
            axis=2
        )

        return np.exp(
            -gamma * dd
        )

    Kxx = ker(X, X)
    Kyy = ker(Y, Y)
    Kxy = ker(X, Y)

    nx = len(X)
    ny = len(Y)

    return float(
        (
            Kxx.sum()
            - np.trace(Kxx)
        ) / max(
            nx * (nx - 1),
            1
        )
        +
        (
            Kyy.sum()
            - np.trace(Kyy)
        ) / max(
            ny * (ny - 1),
            1
        )
        -
        2 * Kxy.mean()
    )


def domain_metrics(syn, real):
    real = real.copy()
    real["domain"] = 1

    alltab = pd.concat(
        [syn, real],
        ignore_index=True,
        sort=False
    )

    X = (
        alltab[DOMAIN_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    y = alltab[
        "domain"
    ].astype(int).to_numpy()

    groups = alltab[
        "site_id"
    ].astype(str).to_numpy()

    gkf = GroupKFold(
        n_splits=5
    )

    probs = np.full(
        len(alltab),
        np.nan
    )

    folds = []

    for fold, (tr, te) in enumerate(
        gkf.split(
            X,
            y,
            groups
        ),
        1
    ):
        if len(
            np.unique(y[te])
        ) < 2:
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000
            )
        )

        clf.fit(
            X.iloc[tr],
            y[tr]
        )

        pr = clf.predict_proba(
            X.iloc[te]
        )[:, 1]

        probs[te] = pr

        folds.append({
            "fold": fold,
            "auc": float(
                roc_auc_score(
                    y[te],
                    pr
                )
            ),
            "n": int(len(te)),
            "sites": int(
                len(
                    np.unique(
                        groups[te]
                    )
                )
            ),
        })

    good = np.isfinite(
        probs
    )

    auc = float(
        roc_auc_score(
            y[good],
            probs[good]
        )
    )

    med = X.median()

    iqr = (
        X.quantile(.75)
        - X.quantile(.25)
    )

    iqr[
        iqr < 1e-6
    ] = 1.0

    Z = (
        X - med
    ) / iqr

    Zs = Z[
        alltab["domain"] == 0
    ].to_numpy(float)

    Zr = Z[
        alltab["domain"] == 1
    ].to_numpy(float)

    mmd = rbf_mmd2(
        Zs,
        Zr
    )

    shifts = []

    for f in DOMAIN_FEATURES:
        a = Z.loc[
            alltab["domain"] == 0,
            f
        ]

        b = Z.loc[
            alltab["domain"] == 1,
            f
        ]

        shifts.append({
            "feature": f,
            "synthetic_median":
                float(syn[f].median()),
            "real_median":
                float(real[f].median()),
            "abs_median_shift_robust_z":
                float(
                    abs(
                        a.median()
                        - b.median()
                    )
                ),
        })

    shifts = pd.DataFrame(
        shifts
    ).sort_values(
        "abs_median_shift_robust_z",
        ascending=False
    )

    return (
        auc,
        mmd,
        pd.DataFrame(folds),
        shifts
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--bridge-dir",
        default="karst_inverse_study/12_domain_bridge"
    )

    ap.add_argument(
        "--cohort",
        default="karst_inverse_study/00_inventory/cohort_core_20y95.csv"
    )

    ap.add_argument(
        "--forcing-dir",
        default="karst_inverse_study/02_forcing_arco/daily"
    )

    ap.add_argument(
        "--out",
        default="karst_inverse_study/14_matrix_conduit_exchange"
    )

    ap.add_argument(
        "--candidates",
        type=int,
        default=700
    )

    ap.add_argument(
        "--rounds",
        type=int,
        default=4
    )

    ap.add_argument(
        "--elite-fraction",
        type=float,
        default=.15
    )

    ap.add_argument(
        "--locked-sims",
        type=int,
        default=700
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260813
    )

    args = ap.parse_args()

    root = Path(
        args.out
    )
    root.mkdir(
        parents=True,
        exist_ok=True
    )

    bridge_dir = Path(
        args.bridge_dir
    )

    bridge = pd.read_csv(
        bridge_dir
        / "real_bridge_descriptors.csv"
    )

    locked = pd.read_csv(
        bridge_dir
        / "real_locked_descriptors.csv"
    )

    with open(
        bridge_dir
        / "bridge_locked_split.json",
        encoding="utf-8"
    ) as f:
        split = json.load(f)

    bridge_sites = list(
        map(
            str,
            split["bridge_sites"]
        )
    )

    locked_sites = list(
        map(
            str,
            split["locked_sites"]
        )
    )

    cohort = pd.read_csv(
        args.cohort
    )

    fmap, startmap = build_forcing_map(
        cohort,
        args.forcing_dir
    )

    print("\nbridge sites:", len(bridge_sites))
    print("locked sites:", len(locked_sites))

    results = {}

    for offset, model_name in enumerate(
        ["M10", "M11"]
    ):
        print("\n" + "=" * 84)

        if model_name == "M10":
            print(
                "M10 — MATRIX -> CONDUIT CASCADE, SAME 7 PARAMETERS"
            )
        else:
            print(
                "M11 — BIDIRECTIONAL MATRIX <-> CONDUIT EXCHANGE"
            )

        print("=" * 84)

        mdir = root / model_name
        mdir.mkdir(
            parents=True,
            exist_ok=True
        )

        proposal = cem_fit(
            model_name,
            bridge,
            bridge_sites,
            fmap,
            startmap,
            args.candidates,
            args.rounds,
            args.elite_fraction,
            args.seed + 1000 * offset,
            mdir,
        )

        print(
            f"\n{model_name} — LOCKED EVALUATION"
        )

        syn = generate_locked(
            model_name,
            proposal,
            fmap,
            startmap,
            locked_sites,
            args.locked_sims,
            args.seed + 5000 + 1000 * offset,
        )

        syn.to_csv(
            mdir
            / "locked_synthetic_descriptors.csv",
            index=False
        )

        auc, mmd, folds, shifts = domain_metrics(
            syn,
            locked
        )

        folds.to_csv(
            mdir
            / "locked_domain_folds.csv",
            index=False
        )

        shifts.to_csv(
            mdir
            / "locked_feature_shifts.csv",
            index=False
        )

        report = {
            "model": model_name,
            "locked_grouped_domain_auc":
                float(auc),
            "locked_rbf_mmd2":
                float(mmd),
            "pass_auc_le_0_75":
                bool(auc <= .75),
            "top_5_shifts":
                shifts.head(5).to_dict(
                    "records"
                ),
        }

        save_json(
            report,
            mdir
            / "locked_domain_gate.json"
        )

        results[
            model_name
        ] = report

        print(
            "locked grouped AUC:",
            round(auc, 3)
        )
        print(
            "locked MMD^2:",
            round(mmd, 4)
        )
        print(
            "PASS:",
            report[
                "pass_auc_le_0_75"
            ]
        )

        print(
            "\nLargest locked shifts:"
        )
        print(
            shifts.head(8).to_string(
                index=False
            )
        )

    if results["M10"]["pass_auc_le_0_75"]:
        decision = (
            "KEEP_M10_CASCADE"
        )
        message = (
            "A matrix-to-conduit cascade closes the locked domain gap "
            "without adding parameters. Rebuild the synthetic/APIC pipeline "
            "around M10."
        )

    elif results["M11"]["pass_auc_le_0_75"]:
        decision = (
            "KEEP_M11_BIDIRECTIONAL_EXCHANGE"
        )
        message = (
            "Unidirectional cascade is insufficient, but dynamic "
            "matrix-conduit exchange closes the locked domain gap. "
            "Rebuild the synthetic/APIC pipeline around M11."
        )

    else:
        decision = (
            "MORE_EXPLICIT_STRUCTURE_REQUIRED"
        )
        message = (
            "Neither cascade nor simple bidirectional exchange closes the "
            "locked domain gap. The next model must represent additional "
            "state dependence / storage-capacity structure rather than "
            "another generic delay."
        )

    summary = {
        "M10": results["M10"],
        "M11": results["M11"],
        "decision": decision,
        "message": message,
    }

    save_json(
        summary,
        root
        / "MATRIX_CONDUIT_DECISION.json"
    )

    print("\n" + "=" * 84)
    print("FINAL DECISION")
    print("=" * 84)
    print(
        "M10 locked AUC:",
        round(
            results["M10"][
                "locked_grouped_domain_auc"
            ],
            3
        )
    )
    print(
        "M11 locked AUC:",
        round(
            results["M11"][
                "locked_grouped_domain_auc"
            ],
            3
        )
    )
    print(
        "DECISION:",
        decision
    )
    print(
        message
    )


if __name__ == "__main__":
    main()
