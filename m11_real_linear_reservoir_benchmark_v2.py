#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m11_real_linear_reservoir_benchmark.py
======================================

Minimal real-data benchmark for the frozen M11 calibration experiment.

A three-parameter single linear reservoir is calibrated on the EXACT SAME
44 springs, central 1825-day windows, forcing and spin-up protocol used for
M11 Stage 1.

Benchmark parameters:
- A_eff_km2: effective discharge scale, log-uniform 0.1--5000 km2
- lambda_et: PET loss factor, uniform 0--1.5
- k_lin: daily linear reservoir release coefficient, log-uniform 1e-4--1 d-1

The benchmark uses the SAME fixed snow/melt constants and a smooth
temperature partition consistent with the M11 setup, so the baseline is
not penalized by completely omitting snow accumulation and melt.

Calibration objective: unweighted SSE in log Q, same as M11.
Evaluation is IN-SAMPLE model adequacy, not predictive validation.

Outputs
-------
karst_inverse_study/31_real_linear_reservoir_benchmark/
    linear_reservoir_fits.csv
    paired_M11_vs_linear.csv
    LINEAR_BENCHMARK_SUMMARY.json
    figures/Figure_M11_vs_linear_benchmark.png

Spyder
------
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/m11_real_linear_reservoir_benchmark.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"
"""

from __future__ import annotations
import argparse,json,math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

import m11_final_apic_stage1 as stage1
import karst_minimal_physics as phys
import m11_real_fingerprint_inference_v2 as realio
import m11_real_exact_inversion_stage1_v2 as exact

T_SNOW=float(phys.T_SNOW_C)
SNOW_W=float(phys.SNOW_TRANSITION_C)
T_MELT=float(phys.T_MELT_C)
DDF=float(phys.DEGREE_DAY_FACTOR_MM_C_DAY)
# 1 mm over 1 km2 = 1000 m3.
M3_PER_MM_KM2=1000.0
SEC_DAY=86400.0


def save_json(x,p):
    Path(p).write_text(json.dumps(x,indent=2),encoding="utf-8")


def sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(x,-60,60)))


def theta_from_u(u):
    u=np.clip(np.asarray(u,float),.001,.999)
    # log ranges for A and k; linear for lambda
    A=np.exp(np.log(.1)+u[0]*(np.log(5000.)-np.log(.1)))
    lam=1.5*u[1]
    k=np.exp(np.log(1e-4)+u[2]*(np.log(1.0)-np.log(1e-4)))
    return np.array([A,lam,k],float)


def lhs(n,seed):
    return .02+.96*qmc.LatinHypercube(d=3,seed=seed).random(n=n)


def simulate_linear(P,T,PET,theta):
    A,lam,k=map(float,theta)
    S=0.0
    SWE=0.0
    q=np.zeros(len(P),float)
    for i in range(len(P)):
        p=max(float(P[i]),0.0)
        tt=float(T[i]); pet=max(float(PET[i]),0.0)
        sf=sigmoid((T_SNOW-tt)/SNOW_W)
        snowfall=p*sf
        rainfall=p-snowfall
        SWE+=snowfall
        melt=min(SWE,DDF*max(tt-T_MELT,0.0))
        SWE-=melt
        recharge=max(0.0,rainfall+melt-lam*pet)
        S+=recharge
        qmm=min(S,k*S)
        S-=qmm
        q[i]=qmm*A*M3_PER_MM_KM2/SEC_DAY
    return q


def build_problem(spin,obs):
    P=np.concatenate([spin["P"].to_numpy(float),obs["P"].to_numpy(float)])
    T=np.concatenate([spin["T"].to_numpy(float),obs["T"].to_numpy(float)])
    PET=np.concatenate([spin["PET"].to_numpy(float),obs["PET"].to_numpy(float)])
    qobs=obs["Q"].to_numpy(float)
    pos=qobs[np.isfinite(qobs)&(qobs>0)]
    floor=max(1e-10,1e-6*float(np.median(pos)) if len(pos) else 1e-10)
    yobs=np.log(np.maximum(qobs,floor))
    nw=len(spin)

    def residual(u):
        q=simulate_linear(P,T,PET,theta_from_u(u))[nw:]
        r=np.log(np.maximum(q,floor))-yobs
        r[~np.isfinite(r)]=100.
        return r
    return P,T,PET,qobs,floor,nw,residual


def fit(fun,nstarts,max_nfev,seed):
    fits=[]
    for x0 in lhs(nstarts,seed):
        r=least_squares(
            fun,x0,bounds=(np.full(3,.001),np.full(3,.999)),
            method="trf",jac="2-point",x_scale="jac",
            max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9
        )
        fits.append((float(2*r.cost),np.asarray(r.x,float),int(r.nfev),bool(r.success)))
    return min(fits,key=lambda z:z[0]),fits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--primary-sites",
        default="karst_inverse_study/16_m11_final_synthetic/locked_primary_supported.csv")
    ap.add_argument("--forcing-dir",default="karst_inverse_study/02_forcing_arco/daily")
    ap.add_argument("--discharge-dir",default="karst_inverse_study/01_discharge")
    ap.add_argument("--m11-out",default="karst_inverse_study/27_m11_real_exact_inversion_v2")
    ap.add_argument("--out",default="karst_inverse_study/31_real_linear_reservoir_benchmark")
    ap.add_argument("--context-days",type=int,default=1825)
    ap.add_argument("--warmup-days",type=int,default=int(stage1.WARMUP_DAYS))
    ap.add_argument("--min-warmup-days",type=int,default=365)
    ap.add_argument("--n-starts",type=int,default=12)
    ap.add_argument("--max-nfev",type=int,default=350)
    ap.add_argument("--seed",type=int,default=20260814)
    a=ap.parse_args()

    out=Path(a.out); figdir=out/"figures"
    out.mkdir(parents=True,exist_ok=True); figdir.mkdir(exist_ok=True)

    sites=pd.read_csv(a.primary_sites)
    col="wokas_id" if "wokas_id" in sites.columns else "site_id"
    sites=sites[col].astype(str).drop_duplicates().tolist()

    # Exact selected windows from M11 run.
    W=pd.read_csv(Path(a.m11_out)/"real_selected_windows.csv")
    W["wokas_id"]=W["wokas_id"].astype(str)
    W=W.set_index("wokas_id")

    rows=[]
    print("\n"+"="*92)
    print("REAL SINGLE-LINEAR-RESERVOIR BENCHMARK")
    print("="*92)
    print(f"sites={len(sites)}; context={a.context_days} d; starts={a.n_starts}")

    for ii,site in enumerate(sites,1):
        print(f"\n{site} ({ii}/{len(sites)})")
        try:
            pf=realio.find_forcing(a.forcing_dir,site)
            pq,dc,qc=realio.find_discharge(a.discharge_dir,site)
            f=realio.load_forcing(pf); q=realio.load_q(pq,dc,qc)
            d,good=realio.merged(q,f)

            if site not in W.index:
                raise RuntimeError("M11 selected window missing")
            wd=W.loc[site]
            start=pd.Timestamp(wd["start_date"])
            end=pd.Timestamp(wd["end_date"])
            obs=d[(d.date>=start)&(d.date<=end)].copy().reset_index(drop=True)
            if len(obs)!=a.context_days:
                raise RuntimeError(f"window length {len(obs)} != {a.context_days}")
            if not np.isfinite(obs[["P","T","PET","Q"]].to_numpy(float)).all():
                raise RuntimeError("non-finite values in benchmark window")

            spin=exact.forcing_spinup(f,start,a.warmup_days)
            if len(spin)<a.min_warmup_days:
                raise RuntimeError(f"only {len(spin)} d spinup")

            P,T,PET,qobs,floor,nw,fun=build_problem(spin,obs)
            best,allfits=fit(fun,a.n_starts,a.max_nfev,a.seed+2003*ii)
            sse,u,nfev,success=best
            theta=theta_from_u(u)
            qsim=simulate_linear(P,T,PET,theta)[nw:]
            dg=exact.diagnostics(qobs,qsim,obs["date"],floor)

            row=dict(
                wokas_id=site,status="ok",
                start_date=str(start.date()),end_date=str(end.date()),
                warmup_days_used=len(spin),
                A_eff_km2=theta[0],lambda_et=theta[1],k_linear_day=theta[2],
                best_sse_log=sse,best_nfev=nfev,best_success=success,
                n_starts=a.n_starts,
                **{k:v for k,v in dg.items()
                   if not k.startswith("obs_") and not k.startswith("sim_")}
            )
            rows.append(row)
            print(f"  logRMSE={dg['rmse_logQ']:.4f}; KGE={dg['KGE']:.3f}; "
                  f"logNSE={dg['logNSE']:.3f}; resACF1={dg['residual_acf1']:.3f}")
        except Exception as e:
            rows.append(dict(wokas_id=site,status="error",error=f"{type(e).__name__}: {e}"))
            print("  ERROR",type(e).__name__,e)

        pd.DataFrame(rows).to_csv(out/"linear_reservoir_fits_PARTIAL.csv",index=False)

    B=pd.DataFrame(rows)
    B.to_csv(out/"linear_reservoir_fits.csv",index=False)

    M=pd.read_csv(Path(a.m11_out)/"real_exact_primary_fits.csv")
    M=M[M.status=="ok"].copy()
    B0=B[B.status=="ok"].copy()
    keep=["wokas_id","rmse_logQ","KGE","NSE","logNSE","residual_acf1",
          "residual_acf7","residual_acf30","percent_bias"]
    P=M[keep].merge(B0[keep],on="wokas_id",suffixes=("_M11","_linear"))
    P["delta_rmse_logQ_linear_minus_M11"]=P.rmse_logQ_linear-P.rmse_logQ_M11
    P["delta_KGE_M11_minus_linear"]=P.KGE_M11-P.KGE_linear
    P["delta_logNSE_M11_minus_linear"]=P.logNSE_M11-P.logNSE_linear
    P["delta_resACF1_linear_minus_M11"]=P.residual_acf1_linear-P.residual_acf1_M11
    P.to_csv(out/"paired_M11_vs_linear.csv",index=False)

    summary={
        "n_paired_sites":int(len(P)),
        "evaluation":"in-sample calibration adequacy, same 1825-d windows",
        "linear_model_parameters":3,
        "M11_parameters":8,
        "median_M11_rmse_logQ":float(P.rmse_logQ_M11.median()),
        "median_linear_rmse_logQ":float(P.rmse_logQ_linear.median()),
        "median_M11_KGE":float(P.KGE_M11.median()),
        "median_linear_KGE":float(P.KGE_linear.median()),
        "median_M11_logNSE":float(P.logNSE_M11.median()),
        "median_linear_logNSE":float(P.logNSE_linear.median()),
        "median_M11_residual_acf1":float(P.residual_acf1_M11.median()),
        "median_linear_residual_acf1":float(P.residual_acf1_linear.median()),
        "fraction_M11_lower_rmse":float((P.rmse_logQ_M11<P.rmse_logQ_linear).mean()),
        "fraction_M11_higher_KGE":float((P.KGE_M11>P.KGE_linear).mean()),
        "median_delta_KGE_M11_minus_linear":float(P.delta_KGE_M11_minus_linear.median()),
        "median_delta_logNSE_M11_minus_linear":float(P.delta_logNSE_M11_minus_linear.median()),
    }
    save_json(summary,out/"LINEAR_BENCHMARK_SUMMARY.json")

    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,5.6))
    ax.scatter(P.KGE_linear,P.KGE_M11)
    lo=min(P.KGE_linear.min(),P.KGE_M11.min())
    hi=max(P.KGE_linear.max(),P.KGE_M11.max())
    ax.plot([lo,hi],[lo,hi],linestyle="--",linewidth=1)
    ax.set_xlabel("Single linear reservoir KGE")
    ax.set_ylabel("M11 KGE")
    fig.tight_layout()
    fig.savefig(figdir/"Figure_M11_vs_linear_benchmark.png",dpi=600,bbox_inches="tight")
    plt.close(fig)

    print("\nLINEAR RESERVOIR BENCHMARK SUMMARY")
    print(json.dumps(summary,indent=2))
    print("\nPAIRED METRICS")
    print(P.to_string(index=False))
    print("\nOutputs:",out)


if __name__=="__main__":
    main()
