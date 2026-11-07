#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m11_revision_required_reruns_v2.py
===============================
Targeted checks requested during the Journal of Hydrology major revision.

Run from the Springs project directory containing the original scripts and
`karst_inverse_study` folder. The script does NOT refit M11. It only:

1) reruns the archived real multistart parameter candidates to evaluate the
   exact stationary-Gaussian AR(1) profile likelihood and reclassify
   near-optimal sets;
2) tests 1-, 2-, 5- and 10-year forcing-only spin-up convergence on the exact
   80 held-out Fisher cases;
3) creates a synthetic hydrograph perturbation figure along the leading and
   weakest Fisher eigenvectors at 90 and 1825 days, using equal prior-whitened
   perturbation magnitude.

Default Spyder command
----------------------
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/m11_revision_required_reruns.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"

Expected existing project outputs
---------------------------------
karst_inverse_study/16_m11_final_synthetic/synthetic_manifest.csv
karst_inverse_study/24_m11_rescue_audit/rescue_audit_cases.csv
karst_inverse_study/24_m11_rescue_audit/case_fisher_matrices.npz
karst_inverse_study/27_m11_real_exact_inversion_v2/real_exact_multistart_runs.csv
karst_inverse_study/27_m11_real_exact_inversion_v2/real_primary_predictions.csv
karst_inverse_study/02_forcing_arco/daily/*.csv
karst_inverse_study/01_discharge/*

Outputs
-------
karst_inverse_study/32_joh_revision_targeted_checks/
  ar1_candidate_relikelihood.csv
  ar1_nearoptimal_site_summary.csv
  ar1_nearoptimal_threshold_sensitivity.csv
  spinup_convergence_casewise.csv
  spinup_convergence_summary.csv
  fisher_direction_case.csv
  Figure_revision_fisher_direction_hydrographs.png
  REVISION_TARGETED_CHECKS_SUMMARY.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

import m11_final_apic_stage1 as stage1
import m11_parameterisation_invariant_rescue_audit as rescue


import re

DATE_NAMES = ["date","Date","DATE","datetime","Datetime","time","Time","timestamp"]
Q_NAMES = [
    "q_clean_m3s","discharge_clean_m3s","Q_clean_m3s","q_m3s","Q_m3s",
    "discharge_m3s","flow_m3s","Q","q","discharge","flow","value_clean","value"
]
P_NAMES = ["precip_clean_mm","precip_mm","tp_mm","precipitation_mm"]
T_NAMES = ["tmean_c","temperature_c","t2m_c","temp_c"]
E_NAMES = ["pet_oudin_mm","pet_mm","PET_mm","pet"]

def safe_id(x):
    return re.sub(r"[^A-Za-z0-9_.()-]+", "_", str(x))

def norm_id(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def firstcol(df, names):
    return next((c for c in names if c in df.columns), None)

def readtab(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table: {path}")

def find_forcing(root, site):
    root = Path(root); target = norm_id(site)
    p = root / f"{safe_id(site)}.csv"
    if p.exists(): return p
    cand = [x for x in root.rglob("*.csv") if norm_id(x.stem) == target]
    if not cand: cand = [x for x in root.rglob("*.csv") if target in norm_id(x.stem)]
    if not cand: raise FileNotFoundError(f"forcing not found for {site} under {root}")
    return sorted(cand, key=lambda x: len(str(x)))[0]

def to_daily_naive(values):
    """Normalize date-like values to timezone-naive calendar days.

    Timezone information is stripped WITHOUT converting the wall-clock time,
    so a daily label such as 2010-01-01 00:00+01:00 remains 2010-01-01.
    This avoids accidental day shifts when merging daily discharge and forcing.
    """
    vals = pd.Series(values)
    out = []
    for x in vals:
        try:
            t = pd.Timestamp(x)
        except Exception:
            t = pd.NaT
        if pd.isna(t):
            out.append(pd.NaT)
            continue
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        out.append(t.normalize())
    return pd.Series(out, index=vals.index, dtype="datetime64[ns]")


def load_forcing(path):
    d = readtab(path)
    dc, pc, tc, ec = firstcol(d, DATE_NAMES), firstcol(d, P_NAMES), firstcol(d, T_NAMES), firstcol(d, E_NAMES)
    if None in [dc, pc, tc, ec]: raise RuntimeError(f"forcing columns not recognized in {path}: {list(d.columns)}")
    z = pd.DataFrame({
        "date": to_daily_naive(d[dc]),
        "P": pd.to_numeric(d[pc], errors="coerce").clip(lower=0),
        "T": pd.to_numeric(d[tc], errors="coerce"),
        "PET": pd.to_numeric(d[ec], errors="coerce"),
    })
    return z.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

def find_discharge(root, site):
    root = Path(root); target = norm_id(site); cand=[]
    for ext in ["*.csv","*.parquet","*.pq"]:
        cand += [p for p in root.rglob(ext) if target in norm_id(p.stem)]
    def score(p):
        s = 100 if norm_id(p.stem)==target else 50
        q=str(p).lower(); s += sum(8 for x in ["clean","audited","processed","daily"] if x in q)
        s -= sum(20 for x in ["summary","manual_review","metadata","inventory"] if x in q)
        return s-.01*len(str(p))
    for p in sorted(cand,key=score,reverse=True):
        try: d=readtab(p)
        except Exception: continue
        dc,qc=firstcol(d,DATE_NAMES),firstcol(d,Q_NAMES)
        if dc is not None and qc is not None: return p,dc,qc
    raise FileNotFoundError(f"discharge not found for {site} under {root}")

def load_q(path, dc, qc):
    d=readtab(path)
    z=pd.DataFrame({"date":to_daily_naive(d[dc]),"Q":pd.to_numeric(d[qc],errors="coerce")})
    z.loc[z.Q<0,"Q"]=np.nan
    return z.dropna(subset=["date"]).drop_duplicates("date",keep="last").sort_values("date").reset_index(drop=True)

def merge_q_forcing(q,f):
    # Defensive re-normalization in case either frame was created elsewhere.
    q=q.copy(); f=f.copy()
    q["date"]=to_daily_naive(q["date"])
    f["date"]=to_daily_naive(f["date"])
    q=q.dropna(subset=["date"]).drop_duplicates("date",keep="last")
    f=f.dropna(subset=["date"]).drop_duplicates("date",keep="last")
    d=q.merge(f,on="date",how="inner",validate="one_to_one").sort_values("date").reset_index(drop=True)
    for c in ["Q","P","T","PET"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d.loc[d.Q<0,"Q"]=np.nan; d.loc[d.P<0,"P"]=0.0
    good=np.isfinite(d[["Q","P","T","PET"]].to_numpy(float)).all(axis=1)
    return d,good

def forcing_spinup(forcing,start_date,requested_days):
    start=pd.Timestamp(start_date).normalize()
    f=forcing[forcing.date<start].copy().reset_index(drop=True)
    if len(f)==0:return f
    arr=f[["P","T","PET"]].to_numpy(float); good=np.isfinite(arr).all(axis=1)
    dd=f.date.to_numpy(dtype="datetime64[D]"); expected=np.datetime64(start.date(),"D")-np.timedelta64(1,"D")
    if dd[-1]!=expected or not good[-1]:return f.iloc[0:0].copy()
    i=len(f)-1
    while i>0 and good[i-1] and dd[i]-dd[i-1]==np.timedelta64(1,"D"):i-=1
    run=f.iloc[i:].copy()
    if len(run)>requested_days:run=run.iloc[-requested_days:].copy()
    return run.reset_index(drop=True)

def theta_from_u(u):
    u=np.clip(np.asarray(u,float),.001,.999)
    return np.array([stage1.q_to_value(p,u[i]) for i,p in enumerate(PARAMS)],float)

def build_problem(spin,obs):
    P=np.concatenate([spin["P"].to_numpy(float),obs["P"].to_numpy(float)])
    T=np.concatenate([spin["T"].to_numpy(float),obs["T"].to_numpy(float)])
    PET=np.concatenate([spin["PET"].to_numpy(float),obs["PET"].to_numpy(float)])
    qobs=obs["Q"].to_numpy(float); pos=qobs[np.isfinite(qobs)&(qobs>0)]
    floor=max(1e-10,1e-6*float(np.median(pos)) if len(pos) else 1e-10)
    yobs=np.log(np.maximum(qobs,floor)); nw=len(spin)
    def residual(u):
        th=theta_from_u(u)
        if not rescue.valid_theta(th):return np.full(len(qobs),100.0)
        sim=rescue.simulate_from_theta(P,T,PET,th)
        if sim is None:return np.full(len(qobs),100.0)
        qsim=sim[0][nw:]; r=np.log(np.maximum(qsim,floor))-yobs; r[~np.isfinite(r)]=100.0
        return r
    return {"P":P,"T":T,"PET":PET,"qobs":qobs,"floor":floor,"nw":nw,"residual":residual}

PARAMS = list(stage1.PARAMS8)
NP = len(PARAMS)
JOINT95 = float(chi2.ppf(.95, df=NP))
DIST_THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.80]
SPINUP_LENGTHS = [365, 730, 1825, 3653]
CONTEXT_DAYS = 1825


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def ar1_profile_objective(resid, rho):
    """
    Profiled stationary Gaussian AR(1) objective (-2 log L up to an
    additive constant independent of rho and of the hydrological candidate).

    With innovation variance tau^2:
        r_1 ~ N(0, tau^2/(1-rho^2))
        r_t = rho r_{t-1} + eps_t, eps_t ~ N(0, tau^2).
    If
        q = (1-rho^2) r_1^2 + sum_{t=2}^n (r_t-rho r_{t-1})^2,
    profiling tau^2=q/n yields
        -2 log L = n log(q/n) - log(1-rho^2) + constant.

    The omitted constant n[log(2*pi)+1] cancels from likelihood differences,
    information-criterion comparisons at fixed n, and profile-likelihood
    near-optimal regions.
    """
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3 or abs(rho) >= 0.9999:
        return np.inf
    one = 1.0 - rho*rho
    q = one*r[0]*r[0] + np.sum((r[1:] - rho*r[:-1])**2)
    if not np.isfinite(q) or q <= 0 or one <= 0:
        return np.inf
    return float(n*math.log(q/n) - math.log(one))

def profile_ar1(resid):
    res = minimize_scalar(
        lambda z: ar1_profile_objective(resid, z),
        bounds=(-0.999, 0.999),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(res.x), float(res.fun)


def max_pairwise_distance(U):
    U = np.asarray(U, float)
    if len(U) < 2:
        return 0.0
    m = 0.0
    for i in range(len(U)-1):
        m = max(m, float(np.linalg.norm(U[i+1:] - U[i], axis=1).max()))
    return m


def u_columns(df):
    cols = [f"u_{p}" for p in PARAMS]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"multistart archive missing columns: {miss}")
    return cols


def real_ar1_nearoptimal(root: Path, out: Path):
    indir = root / "27_m11_real_exact_inversion_v2"
    runs = pd.read_csv(indir / "real_exact_multistart_runs.csv")
    pred = pd.read_csv(indir / "real_primary_predictions.csv")
    runs["wokas_id"] = runs["wokas_id"].astype(str)
    pred["wokas_id"] = pred["wokas_id"].astype(str)
    pred["date"] = to_daily_naive(pred["date"])
    ucols = u_columns(runs)

    cand_rows = []
    site_rows = []
    threshold_rows = []

    sites = sorted(runs["wokas_id"].unique())
    for ii, site in enumerate(sites, 1):
        print(f"[AR1 candidates] {site} ({ii}/{len(sites)})")
        rr = runs[runs.wokas_id == site].copy().reset_index(drop=True)
        pp = pred[pred.wokas_id == site].sort_values("date")
        if len(pp) != CONTEXT_DAYS:
            raise RuntimeError(f"{site}: predictions contain {len(pp)} rows, expected {CONTEXT_DAYS}")
        start = pd.Timestamp(pp.date.min()).normalize()
        end = pd.Timestamp(pp.date.max()).normalize()

        pf = find_forcing(root / "02_forcing_arco" / "daily", site)
        pq, dc, qc = find_discharge(root / "01_discharge", site)
        f = load_forcing(pf)
        q = load_q(pq, dc, qc)
        d, good = merge_q_forcing(q, f)
        obs = d[(d.date >= start) & (d.date <= end)].copy().reset_index(drop=True)
        if len(obs) != CONTEXT_DAYS:
            raise RuntimeError(f"{site}: reconstructed selected window has {len(obs)} rows")
        spin = forcing_spinup(f, start, int(stage1.WARMUP_DAYS))
        if len(spin) < 365:
            raise RuntimeError(f"{site}: only {len(spin)} forcing spin-up days")
        pb = build_problem(spin, obs)

        objs = []
        rhos = []
        for j, row in rr.iterrows():
            u = row[ucols].to_numpy(float)
            resid = pb["residual"](u)
            rho, obj = profile_ar1(resid)
            objs.append(obj); rhos.append(rho)
            cand_rows.append({
                "wokas_id": site,
                "start_index": int(row.start_index),
                "retry_round": int(row.retry_round),
                "sse_log_iid": float(row.sse_log),
                "delta_deviance_iid": float(row.delta_deviance),
                "rho_AR1": rho,
                "minus2logL_AR1_profile": obj,
                **{c: float(row[c]) for c in ucols},
            })

        rr["obj_ar1"] = objs
        rr["rho_ar1"] = rhos
        best_obj = float(rr.obj_ar1.min())
        rr["delta_deviance_ar1"] = rr.obj_ar1 - best_obj
        rr["near_ar1"] = rr.delta_deviance_ar1 <= JOINT95
        near = rr[rr.near_ar1]
        U = near[ucols].to_numpy(float)
        md = max_pairwise_distance(U)
        best_row = rr.loc[rr.obj_ar1.idxmin()]
        iid_best = rr.loc[rr.sse_log.idxmin()]

        site_rows.append({
            "wokas_id": site,
            "n_candidates": int(len(rr)),
            "n_near_AR1": int(len(near)),
            "max_nearoptimal_u_distance_AR1": md,
            "distinct_nearoptimal_AR1_gt_0_45": bool(md > 0.45),
            "best_AR1_start_index": int(best_row.start_index),
            "best_AR1_rho": float(best_row.rho_ar1),
            "best_AR1_minus2logL": float(best_row.obj_ar1),
            "iid_best_start_index": int(iid_best.start_index),
            "same_best_candidate_iid_AR1": bool(int(best_row.start_index) == int(iid_best.start_index)),
        })

        for th in DIST_THRESHOLDS:
            threshold_rows.append({
                "wokas_id": site,
                "distance_threshold": th,
                "distinct_nearoptimal_AR1": bool(md > th),
            })

        # Progressive checkpoints: useful if a later site exposes a data-format issue.
        pd.DataFrame(cand_rows).to_csv(out / "ar1_candidate_relikelihood_PARTIAL.csv", index=False)
        pd.DataFrame(site_rows).to_csv(out / "ar1_nearoptimal_site_summary_PARTIAL.csv", index=False)
        pd.DataFrame(threshold_rows).to_csv(out / "ar1_nearoptimal_threshold_rows_PARTIAL.csv", index=False)

    C = pd.DataFrame(cand_rows)
    S = pd.DataFrame(site_rows)
    T = pd.DataFrame(threshold_rows)
    C.to_csv(out / "ar1_candidate_relikelihood.csv", index=False)
    S.to_csv(out / "ar1_nearoptimal_site_summary.csv", index=False)
    boundary_frac = float((S["best_AR1_rho"].abs() > 0.9985).mean())
    if boundary_frac > 0.25:
        raise RuntimeError(
            f"AR(1) diagnostic failure: {boundary_frac:.1%} of site optima hit |rho|>0.9985. "
            "Check likelihood parameterisation before interpreting results."
        )
    (T.groupby("distance_threshold", as_index=False)
       .agg(n_sites=("wokas_id", "nunique"),
            fraction_distinct_AR1=("distinct_nearoptimal_AR1", "mean"))
       .to_csv(out / "ar1_nearoptimal_threshold_sensitivity.csv", index=False))

    return {
        "n_sites": int(S.wokas_id.nunique()),
        "fraction_distinct_AR1_u_gt_0_45": float(S.distinct_nearoptimal_AR1_gt_0_45.mean()),
        "median_max_distance_AR1": float(S.max_nearoptimal_u_distance_AR1.median()),
        "fraction_same_best_candidate_iid_AR1": float(S.same_best_candidate_iid_AR1.mean()),
        "median_best_candidate_rho_AR1": float(S.best_AR1_rho.median()),
        "fraction_AR1_optima_absrho_gt_0_9985": boundary_frac,
    }


def theta_from_manifest_row(row):
    return np.array([float(row[f"theta_{p}"]) for p in PARAMS], float)


def safe_logdiff(a, b, floor=1e-12):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.log(np.maximum(a, floor)) - np.log(np.maximum(b, floor))


def spinup_convergence(root: Path, out: Path):
    audit = root / "24_m11_rescue_audit"
    cases_path = audit / "rescue_audit_cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError(f"missing {cases_path}")
    cases = pd.read_csv(cases_path)
    man = pd.read_csv(root / "16_m11_final_synthetic" / "synthetic_manifest.csv")
    man = man.set_index("case_id")

    rows = []
    for ii, row in enumerate(cases.itertuples(index=False), 1):
        cid = str(row.case_id)
        print(f"[spin-up] {cid} ({ii}/{len(cases)})")
        if cid not in man.index:
            raise RuntimeError(f"{cid} absent from manifest")
        mr = man.loc[cid]
        casefile = Path(str(row.file))
        if not casefile.exists():
            casefile = Path(str(mr["file"]))
        if not casefile.exists():
            raise FileNotFoundError(f"case NPZ not found for {cid}: {casefile}")
        with np.load(casefile) as z:
            P = np.asarray(z["precip_all_mm"], float)
            T = np.asarray(z["tmean_all_c"], float)
            PET = np.asarray(z["pet_all_mm"], float)
            warmup = int(np.asarray(z["warmup_days"]).ravel()[0])
        theta = theta_from_manifest_row(mr)
        if warmup < max(SPINUP_LENGTHS):
            raise RuntimeError(f"{cid}: archived warmup={warmup} < {max(SPINUP_LENGTHS)}")

        # 10-year reference: exact archived forcing sequence.
        ref_start = warmup - 3653
        ref_end = warmup + CONTEXT_DAYS
        ref = rescue.simulate_from_theta(P[ref_start:ref_end], T[ref_start:ref_end], PET[ref_start:ref_end], theta)
        if ref is None:
            raise RuntimeError(f"{cid}: reference simulation failed")
        qref, smref, scref, _, _ = ref
        ref_boundary = 3653 - 1
        qref_ctx = qref[3653:]
        sm10 = float(smref[ref_boundary]); sc10 = float(scref[ref_boundary])

        for L in SPINUP_LENGTHS:
            st = warmup - L
            en = warmup + CONTEXT_DAYS
            sim = rescue.simulate_from_theta(P[st:en], T[st:en], PET[st:en], theta)
            if sim is None:
                raise RuntimeError(f"{cid}: simulation failed at spin-up {L}")
            q, sm, sc, _, _ = sim
            b = L - 1
            qctx = q[L:]
            dlq = safe_logdiff(qctx, qref_ctx)
            rows.append({
                "case_id": cid,
                "scenario": str(mr["scenario"]),
                "forcing_wokas_id": str(mr["forcing_wokas_id"]),
                "spinup_days": L,
                "abs_Sm_start_diff_mm": abs(float(sm[b]) - sm10),
                "abs_Sc_start_diff_mm": abs(float(sc[b]) - sc10),
                "rel_Sm_start_diff": abs(float(sm[b]) - sm10) / max(abs(sm10), 1.0),
                "rel_Sc_start_diff": abs(float(sc[b]) - sc10) / max(abs(sc10), 1.0),
                "abs_logQ_day1_diff": abs(float(dlq[0])),
                "logQ_RMSE_first30d_vs_10y": float(np.sqrt(np.mean(dlq[:30]**2))),
                "logQ_RMSE_1825d_vs_10y": float(np.sqrt(np.mean(dlq**2))),
            })

    D = pd.DataFrame(rows)
    D.to_csv(out / "spinup_convergence_casewise.csv", index=False)
    summaries = []
    metrics = [
        "abs_Sm_start_diff_mm", "abs_Sc_start_diff_mm",
        "rel_Sm_start_diff", "rel_Sc_start_diff",
        "abs_logQ_day1_diff", "logQ_RMSE_first30d_vs_10y",
        "logQ_RMSE_1825d_vs_10y",
    ]
    for L, g in D.groupby("spinup_days"):
        rec = {"spinup_days": int(L), "n_cases": int(len(g))}
        for m in metrics:
            rec[f"median_{m}"] = float(g[m].median())
            rec[f"p95_{m}"] = float(g[m].quantile(.95))
            rec[f"max_{m}"] = float(g[m].max())
        summaries.append(rec)
    S = pd.DataFrame(summaries).sort_values("spinup_days")
    S.to_csv(out / "spinup_convergence_summary.csv", index=False)
    return {
        "n_cases": int(D.case_id.nunique()),
        "spinup_lengths_days": SPINUP_LENGTHS,
        "summary": S.to_dict("records"),
    }


def fisher_direction_figure(root: Path, out: Path, amplitude=0.5):
    audit = root / "24_m11_rescue_audit"
    cases = pd.read_csv(audit / "rescue_audit_cases.csv")
    Fz = np.load(audit / "case_fisher_matrices.npz")
    man = pd.read_csv(root / "16_m11_final_synthetic" / "synthetic_manifest.csv")
    train = man[(man["split"] == "train") & man["scenario"].isin(["S0", "S1"])].copy()
    man_idx = man.set_index("case_id")

    # Choose case closest to median five-year d_inf, avoiding cherry-picking by fit.
    F5 = np.asarray(Fz["original_1825d"], float)
    dinf = []
    for F in F5:
        lam = np.linalg.eigvalsh(0.5*(F+F.T))
        lam = np.clip(lam, 0, None)
        dinf.append(np.sum(lam/(1+lam)))
    med = np.median(dinf)
    idx = int(np.argmin(np.abs(np.asarray(dinf)-med)))
    cr = cases.iloc[idx]
    cid = str(cr.case_id)
    mr = man_idx.loc[cid]
    theta0 = theta_from_manifest_row(mr)

    # Original transformed coordinates; reference storage arguments are unused.
    stats = rescue.coordinate_stats(train, "original", 1.0, 1.0)
    L = np.asarray(stats["chol"], float)
    c0 = rescue.transformed_coordinates(theta0, "original", 1.0, 1.0)

    casefile = Path(str(cr.file))
    if not casefile.exists(): casefile = Path(str(mr["file"]))
    with np.load(casefile) as z:
        P = np.asarray(z["precip_all_mm"], float)
        T = np.asarray(z["tmean_all_c"], float)
        PET = np.asarray(z["pet_all_mm"], float)
        warmup = int(np.asarray(z["warmup_days"]).ravel()[0])
        dates = np.asarray(z["date"]).astype("datetime64[D]")

    def perturb_theta(v, sign):
        a = float(amplitude)
        while a >= 0.01:
            c = c0 + sign*a*(L @ v)
            th = rescue.coordinates_to_theta(c, "original", 1.0, 1.0)
            if rescue.valid_theta(th):
                return th, a
            a *= 0.75
        raise RuntimeError("Could not find valid eigenvector perturbation")

    # Full baseline simulation once.
    base = rescue.simulate_from_theta(P, T, PET, theta0)[0][warmup:warmup+CONTEXT_DAYS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    metadata = []
    for rowi, days in enumerate([90, 1825]):
        F = np.asarray(Fz[f"original_{days}d"][idx], float)
        w, V = np.linalg.eigh(0.5*(F+F.T))
        order = np.argsort(w)[::-1]
        w = w[order]; V = V[:,order]
        directions = [("leading", V[:,0], w[0]), ("weakest", V[:,-1], w[-1])]
        for coli, (name, v, eig) in enumerate(directions):
            ax = axes[rowi, coli]
            thp, ap = perturb_theta(v, +1)
            thm, am = perturb_theta(v, -1)
            qp = rescue.simulate_from_theta(P, T, PET, thp)[0][warmup:warmup+days]
            qm = rescue.simulate_from_theta(P, T, PET, thm)[0][warmup:warmup+days]
            q0 = base[:days]
            x = np.arange(days)
            ax.plot(x, q0, lw=1.5, label="baseline")
            ax.plot(x, qp, lw=1.0, label=f"+ direction")
            ax.plot(x, qm, lw=1.0, label=f"- direction")
            ax.set_title(f"{days} d - {name} Fisher direction")
            ax.set_xlabel("Day")
            ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
            if rowi == 0 and coli == 0: ax.legend(frameon=False, fontsize=8)
            metadata.append({
                "case_id": cid, "context_days": days, "direction": name,
                "eigenvalue": float(eig), "requested_whitened_amplitude": amplitude,
                "used_plus_amplitude": ap, "used_minus_amplitude": am,
            })
    fig.suptitle(f"Matched synthetic case {cid}: equal prior-whitened perturbations")
    fig.tight_layout()
    fig.savefig(out / "Figure_revision_fisher_direction_hydrographs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metadata).to_csv(out / "fisher_direction_case.csv", index=False)
    return {"case_id": cid, "median_1825d_dinf_case": float(dinf[idx]), "amplitude": amplitude}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="karst_inverse_study")
    ap.add_argument("--out", default="karst_inverse_study/32_joh_revision_targeted_checks")
    ap.add_argument("--skip-real-ar1", action="store_true")
    ap.add_argument("--skip-spinup", action="store_true")
    ap.add_argument("--skip-fisher-figure", action="store_true")
    ap.add_argument("--fisher-amplitude", type=float, default=0.5,
                    help="Prior-whitened Mahalanobis perturbation magnitude")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "REVISION_TARGETED_CHECKS_SUMMARY.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    else:
        summary = {}
    if not args.skip_real_ar1:
        summary["real_AR1_nearoptimal"] = real_ar1_nearoptimal(root, out)
    if not args.skip_spinup:
        summary["spinup_convergence"] = spinup_convergence(root, out)
    if not args.skip_fisher_figure:
        summary["fisher_direction_figure"] = fisher_direction_figure(root, out, args.fisher_amplitude)
    save_json(summary, summary_path)
    print("\n" + "="*80)
    print("DONE. Outputs written to", out.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
