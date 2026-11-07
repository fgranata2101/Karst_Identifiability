#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m11_final_apic_stage1.py
========================

FINAL STAGE-1 PIPELINE AFTER SUPPORT-COVERAGE DECISION.

This script freezes M11 as the physical backbone and prepares the final
synthetic population for the APIC rebuild.

It does NOT test new physical topologies.

What it does
------------
1) Reads the existing M11 CEM population proposal.
2) Reads the existing M11 support-coverage result.
3) Creates explicit real-data cohorts:
      - BRIDGE development sites (25)
      - LOCKED supported primary sites
      - LOCKED OOD secondary sites
4) Creates a FINAL APIC training proposal:
      80% broadened M11 empirical proposal
      20% broad design prior
   to improve JOINT support while retaining realistic density.
5) Splits forcing sites BEFORE simulation into train/val/test.
6) Generates S0-S3 synthetic cases using M11:
      S0: correct M11 physics, core proposal
      S1: correct M11 physics, broadened/tail population
      S2: M11 + omitted state-dependent matrix-capacity/overflow physics
      S3: correct M11 under hidden allogenic precipitation
7) S3 donors are selected ONLY from the SAME split as the base forcing site.
   Therefore base and donor climate sources cannot leak across train/val/test.
8) Saves full forcing prehistory + experiment and all 8 true parameters.

No real discharge is used as a pointwise target in synthetic generation.
The BRIDGE cohort only influenced the already-existing empirical population
proposal; the 52 LOCKED sites remain confirmatory.

Default size
------------
1000 cases/scenario = 4000 total.

Spyder
------
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/m11_final_apic_stage1.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"

Smaller smoke test
------------------
%runfile ".../m11_final_apic_stage1.py" --args "--n-per-scenario 100" --wdir "..."
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

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


WARMUP_DAYS = 10 * 365 + 3
EXPERIMENT_DAYS = 5 * 365 + 2
TOTAL_DAYS = WARMUP_DAYS + EXPERIMENT_DAYS

PARAMS7 = list(PRIOR_BOUNDS)
PARAMS8 = PARAMS7 + ["rho_mc"]

RHO_BOUNDS = ("loguniform", 0.05, 20.0)

LOG_Q_NOISE_SD = 0.02

# Final training-design proposal.
CORE_WEIGHT = 0.80
BROAD_WEIGHT = 0.20
COV_INFLATION = 1.75


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


def load_forcing(path):
    d = pd.read_csv(path)
    d["date"] = civil_date(d["date"])
    d = (
        d.dropna(subset=["date"])
        .sort_values("date")
        .set_index("date")
    )

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

    cs = np.concatenate([[0], np.cumsum(good.astype(int))])
    counts = cs[length:] - cs[:-length]

    return np.flatnonzero(counts == length)


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


def value_to_u(name, x):
    if name == "rho_mc":
        fam, lo, hi = RHO_BOUNDS
    else:
        fam, lo, hi = PRIOR_BOUNDS[name]

    x = float(np.clip(x, lo, hi))

    if fam == "uniform":
        return (x - lo) / (hi - lo)

    if fam == "loguniform":
        return (
            math.log(x) - math.log(lo)
        ) / (
            math.log(hi) - math.log(lo)
        )

    raise ValueError(fam)


def logit_u(u):
    u = np.clip(np.asarray(u, float), 1e-5, 1 - 1e-5)
    return np.log(u / (1 - u))


def sigmoid_z(z):
    z = np.clip(np.asarray(z, float), -12, 12)
    return 1.0 / (1.0 + np.exp(-z))


def params_from_u(u):
    vals = {
        name: q_to_value(name, u[i])
        for i, name in enumerate(PARAMS8)
    }

    p = KarstParams(**{
        k: vals[k]
        for k in PARAMS7
    })

    return p, vals["rho_mc"], vals


def valid_u(u):
    p, rho, vals = params_from_u(u)
    return p.k_fast > p.k_slow


def sample_broad_u(rng):
    for _ in range(10000):
        u = rng.uniform(.01, .99, size=len(PARAMS8))
        if valid_u(u):
            return u
    raise RuntimeError("Could not draw valid broad M11 parameters.")


def sample_empirical_u(rng, mean, cov, inflation=1.0):
    C = np.asarray(cov, float) * float(inflation)

    for _ in range(10000):
        z = rng.multivariate_normal(
            np.asarray(mean, float),
            C,
        )
        u = np.clip(sigmoid_z(z), .005, .995)

        if valid_u(u):
            return u

    raise RuntimeError("Could not draw valid empirical M11 parameters.")


def sample_training_u(rng, mean, cov, scenario):
    """
    S0 focuses on the realistic empirical core.
    S1 deliberately broadens population heterogeneity but retains exact M11.
    S2/S3 use the same final 80/20 design mixture.
    """
    if scenario == "S0":
        if rng.random() < .95:
            return sample_empirical_u(
                rng, mean, cov,
                inflation=COV_INFLATION
            ), "empirical_core"
        return sample_broad_u(rng), "broad_tail"

    if scenario == "S1":
        r = rng.random()
        if r < .60:
            return sample_empirical_u(
                rng, mean, cov,
                inflation=COV_INFLATION * 2.0
            ), "empirical_wide"
        return sample_broad_u(rng), "broad_population"

    if rng.random() < CORE_WEIGHT:
        return sample_empirical_u(
            rng, mean, cov,
            inflation=COV_INFLATION
        ), "empirical_core"

    return sample_broad_u(rng), "broad_tail"


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


def simulate_m11(forcing, p, rho_mc, richer=None):
    """
    M11 truth / fitted backbone.

    If richer is supplied (S2), two omitted processes are activated:
      1) finite matrix capacity with event overflow to conduit;
      2) state-dependent exchange acceleration at high conduit storage.

    These are deliberately absent from the fitted M11 backbone.
    """
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

    Sm = 0.0
    Sc = 0.0
    SWE = 0.0

    q = np.full(n, np.nan)
    Sm_hist = np.full(n, np.nan)
    Sc_hist = np.full(n, np.nan)
    SWE_hist = np.full(n, np.nan)
    R_hist = np.full(n, np.nan)
    phi_hist = np.full(n, np.nan)
    exchange_hist = np.full(n, np.nan)
    overflow_hist = np.zeros(n, dtype=float)

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

        overflow = 0.0

        if richer is not None:
            capacity = richer["matrix_capacity_mm"]

            if Sm > capacity:
                excess = Sm - capacity
                overflow = min(
                    excess,
                    richer["overflow_fraction"] * excess
                )
                Sm -= overflow
                Sc += overflow

        delta = Sm - rho_mc * Sc

        kx = p.k_slow

        if richer is not None:
            scale = (
                1.0
                + richer["exchange_boost"]
                * Sc
                / (Sc + richer["exchange_halfsat_mm"])
            )
            kx = min(1.0, kx * scale)

        mag = (
            kx
            * abs(delta)**p.alpha_slow
        )

        if delta >= 0:
            X = min(Sm, mag)
            Sm -= X
            Sc += X
            signed_X = X
        else:
            X = min(Sc, mag)
            Sc -= X
            Sm += X
            signed_X = -X

        qspring_mm = min(
            Sc,
            p.k_fast
            * max(Sc, 0.0)**p.alpha_fast
        )

        Sc -= qspring_mm

        q[i] = (
            qspring_mm
            * p.A_eff_km2
            * M3_PER_MM_KM2
            / SECONDS_PER_DAY
        )

        Sm_hist[i] = Sm
        Sc_hist[i] = Sc
        SWE_hist[i] = SWE
        R_hist[i] = R
        phi_hist[i] = phi
        exchange_hist[i] = signed_X
        overflow_hist[i] = overflow

    return {
        "q_m3s": q,
        "S_matrix_mm": Sm_hist,
        "S_conduit_mm": Sc_hist,
        "SWE_mm": SWE_hist,
        "effective_recharge_mm": R_hist,
        "fast_fraction": phi_hist,
        "exchange_mm": exchange_hist,
        "overflow_mm": overflow_hist,
    }


def s2_params(rng):
    return {
        "matrix_capacity_mm": float(
            np.exp(
                rng.uniform(
                    np.log(30.0),
                    np.log(400.0)
                )
            )
        ),
        "overflow_fraction": float(
            rng.uniform(.20, .90)
        ),
        "exchange_boost": float(
            rng.uniform(.5, 5.0)
        ),
        "exchange_halfsat_mm": float(
            np.exp(
                rng.uniform(
                    np.log(5.0),
                    np.log(150.0)
                )
            )
        ),
    }


def add_log_noise(q, rng):
    q = np.asarray(q, float)
    out = q.copy()

    good = np.isfinite(q) & (q >= 0)

    if good.any():
        out[good] = (
            q[good]
            * np.exp(
                rng.normal(
                    0.0,
                    LOG_Q_NOISE_SD,
                    good.sum()
                )
            )
        )

    return out


def prepare_s3_true_forcing(
    base,
    donor,
    lag_days,
    allogenic_scale,
):
    donor_p = donor[
        "precip_clean_mm"
    ].to_numpy(float)

    donor_shifted = np.roll(
        donor_p,
        int(lag_days)
    )

    donor_shifted[
        :int(lag_days)
    ] = 0.0

    true = base.copy()

    true[
        "precip_clean_mm"
    ] = (
        base[
            "precip_clean_mm"
        ].to_numpy(float)
        +
        float(allogenic_scale)
        * donor_shifted
    )

    return true


def country_stratified_split(site_ids, seed):
    rng = np.random.default_rng(seed)

    tab = pd.DataFrame({
        "site_id": sorted(set(map(str, site_ids)))
    })

    tab["country"] = (
        tab["site_id"]
        .str.split("-")
        .str[0]
    )

    assign = {}

    for country, g in tab.groupby("country"):
        arr = g["site_id"].to_numpy().copy()
        rng.shuffle(arr)

        n = len(arr)

        ntr = max(1, int(round(.70 * n)))
        nva = max(1, int(round(.15 * n)))

        if ntr + nva >= n:
            nva = max(0, n - ntr - 1)

        tr = arr[:ntr]
        va = arr[ntr:ntr+nva]
        te = arr[ntr+nva:]

        for x in tr:
            assign[x] = "train"
        for x in va:
            assign[x] = "val"
        for x in te:
            assign[x] = "test"

    return assign


def choose_base_window(rng, forcing_map, starts_map, split_sites):
    eligible = [
        wid for wid in split_sites
        if wid in forcing_map
        and len(starts_map.get(wid, []))
    ]

    if not eligible:
        raise RuntimeError(
            "No eligible forcing site in requested split."
        )

    wid = eligible[
        int(rng.integers(0, len(eligible)))
    ]

    starts = starts_map[wid]

    st = int(
        starts[
            int(rng.integers(0, len(starts)))
        ]
    )

    return (
        wid,
        forcing_map[wid].iloc[
            st:st+TOTAL_DAYS
        ].copy()
    )


def choose_donor(
    rng,
    forcing_map,
    starts_map,
    split_sites,
    base_wid,
):
    candidates = [
        wid for wid in split_sites
        if wid != base_wid
        and wid in forcing_map
        and len(starts_map.get(wid, []))
    ]

    if not candidates:
        raise RuntimeError(
            f"No donor available in split for {base_wid}"
        )

    donor_wid = candidates[
        int(rng.integers(0, len(candidates)))
    ]

    starts = starts_map[donor_wid]

    st = int(
        starts[
            int(rng.integers(0, len(starts)))
        ]
    )

    donor = forcing_map[
        donor_wid
    ].iloc[
        st:st+TOTAL_DAYS
    ].copy()

    return donor_wid, donor


def create_real_cohorts(
    bridge_dir,
    support_dir,
    out,
):
    with open(
        Path(bridge_dir)
        / "bridge_locked_split.json",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    site_support = pd.read_csv(
        Path(support_dir)
        / "M11"
        / "real_site_support.csv"
    )

    bridge = pd.DataFrame({
        "wokas_id": list(
            map(str, split["bridge_sites"])
        ),
        "analysis_role": "bridge_development",
    })

    locked_supported = (
        site_support[
            site_support["site_supported"].astype(bool)
        ][["site_id"]]
        .rename(columns={"site_id": "wokas_id"})
    )
    locked_supported[
        "analysis_role"
    ] = "locked_primary_supported"

    locked_ood = (
        site_support[
            ~site_support["site_supported"].astype(bool)
        ][["site_id"]]
        .rename(columns={"site_id": "wokas_id"})
    )
    locked_ood[
        "analysis_role"
    ] = "locked_secondary_ood"

    all_roles = pd.concat(
        [
            bridge,
            locked_supported,
            locked_ood,
        ],
        ignore_index=True
    )

    all_roles.to_csv(
        out / "real_analysis_cohorts.csv",
        index=False
    )

    locked_supported.to_csv(
        out / "locked_primary_supported.csv",
        index=False
    )

    locked_ood.to_csv(
        out / "locked_secondary_ood.csv",
        index=False
    )

    return {
        "n_bridge":
            int(len(bridge)),
        "n_locked_primary_supported":
            int(len(locked_supported)),
        "n_locked_secondary_ood":
            int(len(locked_ood)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--cohort",
        default=(
            "karst_inverse_study/00_inventory/"
            "cohort_core_20y95.csv"
        ),
    )

    ap.add_argument(
        "--forcing-dir",
        default=(
            "karst_inverse_study/"
            "02_forcing_arco/daily"
        ),
    )

    ap.add_argument(
        "--m11-prior",
        default=(
            "karst_inverse_study/"
            "14_matrix_conduit_exchange/"
            "M11/M11_population_prior.json"
        ),
    )

    ap.add_argument(
        "--bridge-dir",
        default=(
            "karst_inverse_study/"
            "12_domain_bridge"
        ),
    )

    ap.add_argument(
        "--support-dir",
        default=(
            "karst_inverse_study/"
            "15_support_coverage"
        ),
    )

    ap.add_argument(
        "--out",
        default=(
            "karst_inverse_study/"
            "16_m11_final_synthetic"
        ),
    )

    ap.add_argument(
        "--n-per-scenario",
        type=int,
        default=1000,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260813,
    )

    args = ap.parse_args()

    root = Path(args.out)
    cases_dir = root / "cases"

    root.mkdir(
        parents=True,
        exist_ok=True
    )
    cases_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cohort_counts = create_real_cohorts(
        args.bridge_dir,
        args.support_dir,
        root,
    )

    with open(
        args.m11_prior,
        encoding="utf-8",
    ) as f:
        proposal = json.load(f)

    mean = np.asarray(
        proposal["logit_u_mean"],
        float
    )

    cov = np.asarray(
        proposal["logit_u_cov"],
        float
    )

    if len(mean) != len(PARAMS8):
        raise RuntimeError(
            f"M11 prior has dimension {len(mean)}, "
            f"expected {len(PARAMS8)}"
        )

    cohort = pd.read_csv(
        args.cohort
    )

    forcing_map = {}
    starts_map = {}

    for wid in cohort[
        "wokas_id"
    ].astype(str):
        p = (
            Path(args.forcing_dir)
            / f"{safe_id(wid)}.csv"
        )

        if not p.exists():
            continue

        d = load_forcing(p)
        starts = contiguous_valid_starts(
            d,
            TOTAL_DAYS
        )

        if len(starts):
            forcing_map[wid] = d
            starts_map[wid] = starts

    if len(forcing_map) < 20:
        raise RuntimeError(
            "Too few forcing sites available."
        )

    site_split = country_stratified_split(
        forcing_map.keys(),
        args.seed,
    )

    split_sites = {
        split: sorted([
            wid for wid, s in site_split.items()
            if s == split
        ])
        for split in ["train", "val", "test"]
    }

    save_json(
        {
            "seed": args.seed,
            "train_sites": split_sites["train"],
            "val_sites": split_sites["val"],
            "test_sites": split_sites["test"],
        },
        root / "forcing_split.json",
    )

    rng = np.random.default_rng(
        args.seed
    )

    rows = []

    target_split_fracs = {
        "train": .70,
        "val": .15,
        "test": .15,
    }

    for scenario in ["S0", "S1", "S2", "S3"]:
        scen_dir = cases_dir / scenario
        scen_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for case_idx in range(
            args.n_per_scenario
        ):
            r = rng.random()

            if r < .70:
                split = "train"
            elif r < .85:
                split = "val"
            else:
                split = "test"

            base_wid, base = choose_base_window(
                rng,
                forcing_map,
                starts_map,
                split_sites[split],
            )

            u, draw_class = sample_training_u(
                rng,
                mean,
                cov,
                scenario,
            )

            p, rho_mc, vals = params_from_u(
                u
            )

            richer = None
            donor_wid = ""
            donor_lag = np.nan
            donor_scale = np.nan

            if scenario == "S2":
                richer = s2_params(rng)
                sim_true = simulate_m11(
                    base,
                    p,
                    rho_mc,
                    richer=richer,
                )

            elif scenario == "S3":
                donor_wid, donor = choose_donor(
                    rng,
                    forcing_map,
                    starts_map,
                    split_sites[split],
                    base_wid,
                )

                donor_lag = int(
                    rng.integers(0, 8)
                )

                donor_scale = float(
                    rng.uniform(.15, 1.25)
                )

                true_forcing = (
                    prepare_s3_true_forcing(
                        base,
                        donor,
                        donor_lag,
                        donor_scale,
                    )
                )

                sim_true = simulate_m11(
                    true_forcing,
                    p,
                    rho_mc,
                    richer=None,
                )

            else:
                sim_true = simulate_m11(
                    base,
                    p,
                    rho_mc,
                    richer=None,
                )

            q_true_all = sim_true[
                "q_m3s"
            ]

            q_obs_all = add_log_noise(
                q_true_all,
                rng
            )

            sl = slice(
                WARMUP_DAYS,
                TOTAL_DAYS
            )

            case_id = (
                f"{scenario}_"
                f"{case_idx:05d}"
            )

            target = (
                scen_dir
                / f"{case_id}.npz"
            )

            p_all = base[
                "precip_clean_mm"
            ].to_numpy(float)

            t_all = base[
                "tmean_c"
            ].to_numpy(float)

            pet_all = base[
                "pet_oudin_mm"
            ].to_numpy(float)

            payload = {
                "date":
                    base.index[sl].to_numpy(
                        dtype="datetime64[D]"
                    ),
                "date_all":
                    base.index.to_numpy(
                        dtype="datetime64[D]"
                    ),
                "precip_obs_mm":
                    p_all[sl].astype(np.float32),
                "tmean_obs_c":
                    t_all[sl].astype(np.float32),
                "pet_obs_mm":
                    pet_all[sl].astype(np.float32),
                "q_true_m3s":
                    q_true_all[sl].astype(np.float32),
                "q_obs_m3s":
                    q_obs_all[sl].astype(np.float32),
                "precip_all_mm":
                    p_all.astype(np.float32),
                "tmean_all_c":
                    t_all.astype(np.float32),
                "pet_all_mm":
                    pet_all.astype(np.float32),
                "q_true_all_m3s":
                    q_true_all.astype(np.float32),
                "warmup_days":
                    np.array(
                        [WARMUP_DAYS],
                        dtype=np.int32
                    ),
            }

            if richer is not None:
                for k, v in richer.items():
                    payload[
                        f"extra_{k}"
                    ] = np.array(
                        [v],
                        dtype=np.float32
                    )

            np.savez_compressed(
                target,
                **payload
            )

            row = {
                "case_id":
                    case_id,
                "scenario":
                    scenario,
                "split":
                    split,
                "forcing_wokas_id":
                    base_wid,
                "donor_wokas_id":
                    donor_wid,
                "draw_class":
                    draw_class,
                "allogenic_lag_days":
                    donor_lag,
                "allogenic_scale":
                    donor_scale,
                "file":
                    str(target.resolve()),
            }

            for i, name in enumerate(PARAMS8):
                row[
                    f"u_{name}"
                ] = float(u[i])
                row[
                    f"theta_{name}"
                ] = float(vals[name])

            if richer is not None:
                for k, v in richer.items():
                    row[
                        f"extra_{k}"
                    ] = float(v)

            rows.append(row)

            if (
                (case_idx + 1) % 100 == 0
            ):
                print(
                    scenario,
                    case_idx + 1,
                    "/",
                    args.n_per_scenario
                )

        pd.DataFrame(
            rows
        ).to_csv(
            root
            / "synthetic_manifest_partial.csv",
            index=False
        )

    man = pd.DataFrame(rows)

    man.to_csv(
        root
        / "synthetic_manifest.csv",
        index=False
    )

    donor_cross_split = 0

    s3 = man[
        man["scenario"] == "S3"
    ]

    for row in s3.itertuples(index=False):
        if (
            site_split.get(
                str(row.donor_wokas_id)
            )
            != row.split
        ):
            donor_cross_split += 1

    design = {
        "backbone": "M11 bidirectional matrix-conduit exchange",
        "parameters": PARAMS8,
        "n_per_scenario": args.n_per_scenario,
        "n_total": int(len(man)),
        "training_design_distribution": {
            "empirical_core_weight": CORE_WEIGHT,
            "broad_design_weight": BROAD_WEIGHT,
            "empirical_covariance_inflation": COV_INFLATION,
        },
        "forcing_split_counts": {
            k: len(v)
            for k, v in split_sites.items()
        },
        "scenario_counts": (
            man["scenario"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "split_counts": (
            man["split"]
            .value_counts()
            .to_dict()
        ),
        "s3_donor_cross_split_cases":
            int(donor_cross_split),
        "real_cohorts":
            cohort_counts,
        "real_discharge_used_as_pointwise_synthetic_target":
            False,
        "note": (
            "BRIDGE real descriptors informed the empirical population proposal. "
            "LOCKED sites remain confirmatory; OOD flags are preserved."
        ),
    }

    save_json(
        design,
        root / "M11_FINAL_SYNTHETIC_DESIGN.json",
    )

    print("\n=== M11 FINAL SYNTHETIC POPULATION ===")
    print(
        man["scenario"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSplit:")
    print(
        man["split"]
        .value_counts()
        .to_string()
    )

    print(
        "\nS3 donor cross-split cases:",
        donor_cross_split
    )

    print(
        "\nReal cohorts:",
        cohort_counts
    )

    print(
        "\nWritten to:",
        root.resolve()
    )


if __name__ == "__main__":
    main()
