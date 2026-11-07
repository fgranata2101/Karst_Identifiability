#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
karst_minimal_physics.py

Minimal, deliberately incomplete two-store karst simulator for the
"Karst Dynamical Grammar" project.

State variables are expressed as equivalent water depth [mm] over an unknown
hydrologically effective area A_eff [km2]. Discharge is converted to m3/s.

The model is deliberately parsimonious. Structural discrepancy is NOT inserted
here; it belongs to the APIC discrepancy branch.

Site-specific parameter vector (7 parameters):
    A_eff_km2      hydrologically effective contributing area / scale
    lambda_et      PET loss multiplier
    k_slow         slow-store drainage coefficient
    alpha_slow     slow-store nonlinearity
    k_fast         fast-store drainage coefficient
    alpha_fast     fast-store nonlinearity
    tau_fast_mm    recharge threshold for fast-path activation

Global fixed snow/gating constants are kept out of site parameters in v1.

No parameter should be interpreted hydrogeologically on real data unless the
synthetic identifiability gate defined in the preregistration is passed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0
M3_PER_MM_KM2 = 1000.0  # 1 mm over 1 km2


@dataclass(frozen=True)
class KarstParams:
    A_eff_km2: float
    lambda_et: float
    k_slow: float
    alpha_slow: float
    k_fast: float
    alpha_fast: float
    tau_fast_mm: float


PRIOR_BOUNDS = {
    # Distribution family and bounds. These are INITIAL synthetic design priors;
    # they must be stress-tested before real-data interpretation.
    "A_eff_km2": ("loguniform", 0.1, 5000.0),
    "lambda_et": ("uniform", 0.0, 1.5),
    "k_slow": ("loguniform", 5e-4, 0.10),
    "alpha_slow": ("uniform", 0.70, 2.00),
    "k_fast": ("loguniform", 0.02, 1.00),
    "alpha_fast": ("uniform", 0.70, 2.50),
    "tau_fast_mm": ("uniform", 0.0, 40.0),
}

# Global constants in v1.
T_SNOW_C = 1.0
SNOW_TRANSITION_C = 1.0
T_MELT_C = 0.0
DEGREE_DAY_FACTOR_MM_C_DAY = 3.0
FAST_GATE_WIDTH_MM = 2.0


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def sample_prior(rng: np.random.Generator, n=1):
    def draw(name):
        fam, lo, hi = PRIOR_BOUNDS[name]
        if fam == "uniform":
            return rng.uniform(lo, hi, n)
        if fam == "loguniform":
            return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        raise ValueError(fam)

    arrays = {k: draw(k) for k in PRIOR_BOUNDS}
    return [
        KarstParams(**{k: float(arrays[k][i]) for k in arrays})
        for i in range(n)
    ]


def validate_params(p: KarstParams):
    for name, (fam, lo, hi) in PRIOR_BOUNDS.items():
        v = getattr(p, name)
        if not (lo <= v <= hi):
            raise ValueError(f"{name}={v} fuori prior [{lo}, {hi}]")
    if p.k_fast <= p.k_slow:
        # Not mathematically mandatory, but the labels "fast" and "slow"
        # would otherwise lose their intended ordering.
        raise ValueError("k_fast deve essere > k_slow in v1.")


def simulate(forcing: pd.DataFrame, p: KarstParams,
             initial_slow_mm=0.0, initial_fast_mm=0.0, initial_swe_mm=0.0):
    """
    forcing columns:
        precip_clean_mm  [mm/day]
        tmean_c          [degC]
        pet_oudin_mm     [mm/day]

    Returns daily states and discharge. Missing forcing produces NaN output
    for that day; states are carried forward without fabricated input.
    """
    validate_params(p)

    req = ("precip_clean_mm", "tmean_c", "pet_oudin_mm")
    missing = [c for c in req if c not in forcing.columns]
    if missing:
        raise KeyError(f"forcing missing columns {missing}")

    idx = pd.DatetimeIndex(forcing.index)
    n = len(forcing)

    Sslow = max(0.0, float(initial_slow_mm))
    Sfast = max(0.0, float(initial_fast_mm))
    SWE = max(0.0, float(initial_swe_mm))

    out = {
        "snowfall_mm": np.full(n, np.nan),
        "rainfall_mm": np.full(n, np.nan),
        "snowmelt_mm": np.full(n, np.nan),
        "effective_recharge_mm": np.full(n, np.nan),
        "fast_fraction": np.full(n, np.nan),
        "q_slow_mm": np.full(n, np.nan),
        "q_fast_mm": np.full(n, np.nan),
        "q_phys_m3s": np.full(n, np.nan),
        "S_slow_mm": np.full(n, np.nan),
        "S_fast_mm": np.full(n, np.nan),
        "SWE_mm": np.full(n, np.nan),
    }

    P = pd.to_numeric(forcing["precip_clean_mm"], errors="coerce").to_numpy(float)
    T = pd.to_numeric(forcing["tmean_c"], errors="coerce").to_numpy(float)
    PET = pd.to_numeric(forcing["pet_oudin_mm"], errors="coerce").to_numpy(float)

    for i in range(n):
        if not (np.isfinite(P[i]) and np.isfinite(T[i]) and np.isfinite(PET[i])):
            # No invented meteorology.
            out["S_slow_mm"][i] = Sslow
            out["S_fast_mm"][i] = Sfast
            out["SWE_mm"][i] = SWE
            continue

        snow_frac = sigmoid((T_SNOW_C - T[i]) / SNOW_TRANSITION_C)
        snowfall = max(0.0, P[i]) * snow_frac
        rainfall = max(0.0, P[i]) - snowfall
        SWE += snowfall

        potential_melt = DEGREE_DAY_FACTOR_MM_C_DAY * max(T[i] - T_MELT_C, 0.0)
        melt = min(SWE, potential_melt)
        SWE -= melt

        liquid = rainfall + melt
        recharge = max(0.0, liquid - p.lambda_et * max(PET[i], 0.0))

        fast_fraction = sigmoid((recharge - p.tau_fast_mm) / FAST_GATE_WIDTH_MM)
        I_fast = fast_fraction * recharge
        I_slow = recharge - I_fast

        Sslow += I_slow
        Sfast += I_fast

        qslow = min(Sslow, p.k_slow * (max(Sslow, 0.0) ** p.alpha_slow))
        qfast = min(Sfast, p.k_fast * (max(Sfast, 0.0) ** p.alpha_fast))
        Sslow -= qslow
        Sfast -= qfast

        q_m3s = (qslow + qfast) * p.A_eff_km2 * M3_PER_MM_KM2 / SECONDS_PER_DAY

        out["snowfall_mm"][i] = snowfall
        out["rainfall_mm"][i] = rainfall
        out["snowmelt_mm"][i] = melt
        out["effective_recharge_mm"][i] = recharge
        out["fast_fraction"][i] = fast_fraction
        out["q_slow_mm"][i] = qslow
        out["q_fast_mm"][i] = qfast
        out["q_phys_m3s"][i] = q_m3s
        out["S_slow_mm"][i] = Sslow
        out["S_fast_mm"][i] = Sfast
        out["SWE_mm"][i] = SWE

    return pd.DataFrame(out, index=idx)


def selftest():
    rng = np.random.default_rng(20260810)
    idx = pd.date_range("2000-01-01", periods=3650, freq="D")
    # Synthetic seasonal forcing.
    doy = idx.dayofyear.to_numpy()
    temp = 8 + 10*np.sin(2*np.pi*(doy-170)/365.25)
    precip = rng.gamma(0.7, 4.0, len(idx))
    precip[rng.random(len(idx)) < 0.65] = 0.0
    pet = np.maximum(0, 2 + 2*np.sin(2*np.pi*(doy-170)/365.25))
    forcing = pd.DataFrame({
        "precip_clean_mm": precip,
        "tmean_c": temp,
        "pet_oudin_mm": pet,
    }, index=idx)

    # Draw until fast > slow.
    for p in sample_prior(rng, 100):
        if p.k_fast > p.k_slow:
            break
    sim = simulate(forcing, p)
    assert (sim[["S_slow_mm", "S_fast_mm", "SWE_mm"]].dropna() >= -1e-12).all().all()
    assert (sim["q_phys_m3s"].dropna() >= 0).all()
    assert len(sim) == len(forcing)
    print("karst_minimal_physics selftest: OK")
    print("example parameters:", asdict(p))
    print("median Q:", float(sim["q_phys_m3s"].median()))
    return 0


if __name__ == "__main__":
    selftest()
