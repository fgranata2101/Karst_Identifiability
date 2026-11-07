#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m11_real_exact_inversion_stage1_v2.py

Stage-1 real-data gate after confirmatory synthetic identifiability:
fit frozen M11 directly to the central complete 1825-d window of all
44 LOCKED-supported real springs.

NO APIC. NO fingerprints. NO Q interpolation. NO model change.

The goal is model adequacy, not yet physical parameter interpretation.
Real residual variance is estimated from each fit; synthetic sigma_logQ=0.02
is reported only as a reference.

Outputs:
karst_inverse_study/27_m11_real_exact_inversion_v2/
  real_exact_primary_fits.csv
  real_exact_multistart_runs.csv
  real_parameter_estimates.csv
  real_predictive_diagnostics.csv
  real_primary_predictions.csv
  real_selected_windows.csv
  REAL_EXACT_INVERSION_STAGE1_SUMMARY.json

Spyder:
%runfile "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs/m11_real_exact_inversion_stage1_v2.py" --wdir "C:/Users/franc/OneDrive/Documenti/Ricerca/Springs"
"""

from __future__ import annotations
import os, json, math, argparse, traceback
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc, chi2

import m11_final_apic_stage1 as stage1
import m11_parameterisation_invariant_rescue_audit as rescue
import m11_real_fingerprint_inference_v2 as realio

PARAMS = list(stage1.PARAMS8)
NP = len(PARAMS)
SYNTH_SIGMA = float(stage1.LOG_Q_NOISE_SD)
JOINT95 = float(chi2.ppf(.95, df=NP))


def save_json(x, p):
    Path(p).write_text(json.dumps(x, indent=2), encoding="utf-8")


def theta_from_u(u):
    u = np.clip(np.asarray(u, float), .001, .999)
    return np.array([stage1.q_to_value(p, u[i]) for i, p in enumerate(PARAMS)], float)


def lhs(n, seed):
    return .02 + .96*qmc.LatinHypercube(d=NP, seed=seed).random(n=n)


def forcing_spinup(forcing, start_date, requested_days):
    start = pd.Timestamp(start_date).normalize()
    f = forcing[forcing.date < start].copy().reset_index(drop=True)
    if len(f) == 0:
        return f
    arr = f[["P","T","PET"]].to_numpy(float)
    good = np.isfinite(arr).all(axis=1)
    dd = f["date"].to_numpy(dtype="datetime64[D]")
    expected = np.datetime64(start.date(),"D") - np.timedelta64(1,"D")
    if dd[-1] != expected or not good[-1]:
        return f.iloc[0:0].copy()
    i = len(f)-1
    while i > 0 and good[i-1] and dd[i]-dd[i-1] == np.timedelta64(1,"D"):
        i -= 1
    run = f.iloc[i:].copy()
    if len(run) > requested_days:
        run = run.iloc[-requested_days:].copy()
    return run.reset_index(drop=True)


def build_problem(spin, obs):
    # Defensive numeric coercion: only "date" may be datetime/object.
    spin = spin.copy()
    obs = obs.copy()
    for c in ["P", "T", "PET"]:
        spin[c] = pd.to_numeric(spin[c], errors="coerce")
    for c in ["P", "T", "PET", "Q"]:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")

    spin_num = spin[["P", "T", "PET"]].to_numpy(dtype=float)
    obs_num = obs[["P", "T", "PET", "Q"]].to_numpy(dtype=float)

    if not np.isfinite(spin_num).all():
        raise RuntimeError("spin-up contains non-finite P/T/PET values")
    if not np.isfinite(obs_num).all():
        raise RuntimeError("observation window contains non-finite P/T/PET/Q values")

    P = np.concatenate([spin["P"].to_numpy(dtype=float), obs["P"].to_numpy(dtype=float)])
    T = np.concatenate([spin["T"].to_numpy(dtype=float), obs["T"].to_numpy(dtype=float)])
    PET = np.concatenate([spin["PET"].to_numpy(dtype=float), obs["PET"].to_numpy(dtype=float)])
    qobs = obs["Q"].to_numpy(dtype=float)
    pos = qobs[np.isfinite(qobs) & (qobs > 0)]
    floor = max(1e-10, 1e-6*float(np.median(pos)) if len(pos) else 1e-10)
    yobs = np.log(np.maximum(qobs, floor))
    nw = len(spin)

    def residual(u):
        th = theta_from_u(u)
        if not rescue.valid_theta(th):
            return np.full(len(qobs), 30.0 + 200.0*max(0.0, th[2]-th[4]))
        sim = rescue.simulate_from_theta(P,T,PET,th)
        if sim is None:
            return np.full(len(qobs), 100.0)
        qsim = sim[0][nw:]
        r = np.log(np.maximum(qsim,floor)) - yobs
        r[~np.isfinite(r)] = 100.0
        return r

    return dict(P=P,T=T,PET=PET,qobs=qobs,yobs=yobs,floor=floor,nw=nw,residual=residual)


def fit_once(fun, x0, max_nfev):
    r = least_squares(
        fun, np.asarray(x0,float),
        bounds=(np.full(NP,.001),np.full(NP,.999)),
        method="trf", jac="2-point", x_scale="jac",
        max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9
    )
    return dict(success=bool(r.success), nfev=int(r.nfev),
                optimality=float(r.optimality), sse=float(2*r.cost),
                u=np.asarray(r.x,float))


def deviance(sse, best, n):
    if not np.isfinite(sse) or not np.isfinite(best) or best <= 0:
        return np.inf
    return float(n*np.log(max(sse/best,1e-300)))


def maxdist(U):
    U=np.asarray(U,float)
    if len(U)<2:return 0.0
    return max(float(np.linalg.norm(U[i+1:]-U[i],axis=1).max())
               for i in range(len(U)-1))


def multistart(fun, nobs, nstarts, max_nfev, retry_starts, retry_nfev, seed):
    fits=[]
    for i,x0 in enumerate(lhs(nstarts,seed),1):
        f=fit_once(fun,x0,max_nfev); f.update(start_index=i,retry_round=0); fits.append(f)

    def finalize(fs):
        b=min(fs,key=lambda z:z["sse"])
        for z in fs:z["delta_deviance"]=deviance(z["sse"],b["sse"],nobs)
        near=[z for z in fs if z["delta_deviance"]<=JOINT95]
        return b,near

    best,near=finalize(fits)
    retry = len(near)<2
    if retry:
        for x0 in lhs(retry_starts,seed+1000003):
            f=fit_once(fun,x0,retry_nfev)
            f.update(start_index=len(fits)+1,retry_round=1); fits.append(f)
        best,near=finalize(fits)

    U=np.vstack([z["u"] for z in near])
    return best,fits,near,maxdist(U),retry


def simulate_best(pb,u):
    th=theta_from_u(u)
    s=rescue.simulate_from_theta(pb["P"],pb["T"],pb["PET"],th)
    if s is None: raise RuntimeError("best simulation failed")
    return th,s[0][pb["nw"]:]


def acf(x,lag):
    s=pd.Series(np.asarray(x,float))
    return float(s.autocorr(lag=lag)) if s.notna().sum()>lag+5 else np.nan


def nse(o,s):
    o=np.asarray(o,float); s=np.asarray(s,float)
    den=np.sum((o-o.mean())**2)
    return float(1-np.sum((s-o)**2)/den) if den>0 else np.nan


def kge(o,s):
    o=np.asarray(o,float); s=np.asarray(s,float)
    if len(o)<3 or np.std(o,ddof=1)<=0 or np.mean(o)==0:return np.nan
    r=np.corrcoef(o,s)[0,1]
    a=np.std(s,ddof=1)/np.std(o,ddof=1); b=np.mean(s)/np.mean(o)
    return float(1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2))


def flash(q):
    q=np.asarray(q,float)
    return float(np.abs(np.diff(q)).sum()/q.sum()) if len(q)>2 and q.sum()>0 else np.nan


def recession(q):
    q=np.asarray(q,float); l=np.where(q>0,np.log(q),np.nan); d=np.diff(l)
    x=-d[np.isfinite(d)&(d<0)]
    return float(np.median(x)) if len(x)>=20 else np.nan


def seasonal(q,dates):
    d=pd.DataFrame({"q":q,"m":pd.DatetimeIndex(dates).month})
    d["l"]=np.log1p(np.maximum(d.q,0))
    x=d.groupby("m").l.mean()
    return float(x.max()-x.min()) if len(x)>=10 else np.nan


def signatures(q,dates):
    q=np.asarray(q,float); l=np.log1p(np.maximum(q,0))
    return dict(q05=float(np.quantile(q,.05)),q50=float(np.quantile(q,.5)),
                q95=float(np.quantile(q,.95)),flashiness=flash(q),
                recession_log_slope=recession(q),
                acf1_logq=acf(l,1),acf7_logq=acf(l,7),acf30_logq=acf(l,30),
                seasonal_logq_amplitude=seasonal(q,dates))


def diagnostics(obs,sim,dates,floor):
    yo=np.log(np.maximum(obs,floor)); ys=np.log(np.maximum(sim,floor)); r=ys-yo
    rm=float(np.sqrt(np.mean(r*r)))
    O=signatures(obs,dates); S=signatures(sim,dates)
    z=dict(rmse_logQ=rm,rmse_logQ_over_synthetic_sigma=rm/SYNTH_SIGMA,
           NSE=nse(obs,sim),logNSE=nse(yo,ys),KGE=kge(obs,sim),
           percent_bias=float(100*(sim.sum()-obs.sum())/max(obs.sum(),1e-30)),
           residual_acf1=acf(r,1),residual_acf7=acf(r,7),residual_acf30=acf(r,30))
    for k,v in O.items(): z["obs_"+k]=v
    for k,v in S.items(): z["sim_"+k]=v
    for k in ["q05","q50","q95"]:
        z["log_ratio_sim_obs_"+k]=float(np.log(max(S[k],floor)/max(O[k],floor)))
    return z


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--primary-sites",default="karst_inverse_study/16_m11_final_synthetic/locked_primary_supported.csv")
    ap.add_argument("--forcing-dir",default="karst_inverse_study/02_forcing_arco/daily")
    ap.add_argument("--discharge-dir",default="karst_inverse_study/01_discharge")
    ap.add_argument("--out",default="karst_inverse_study/27_m11_real_exact_inversion_v2")
    ap.add_argument("--context-days",type=int,default=1825)
    ap.add_argument("--warmup-days",type=int,default=int(stage1.WARMUP_DAYS))
    ap.add_argument("--min-warmup-days",type=int,default=365)
    ap.add_argument("--n-starts",type=int,default=12)
    ap.add_argument("--max-nfev",type=int,default=350)
    ap.add_argument("--retry-starts",type=int,default=24)
    ap.add_argument("--retry-max-nfev",type=int,default=700)
    ap.add_argument("--seed",type=int,default=20260814)
    a=ap.parse_args()

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(a.primary_sites); col="wokas_id" if "wokas_id" in s.columns else "site_id"
    sites=s[col].astype(str).drop_duplicates().tolist()

    FIT=[]; RUN=[]; PAR=[]; DIA=[]; PRED=[]; WIN=[]

    print("\n"+"="*92)
    print("M11 REAL EXACT INVERSION — STAGE 1 MODEL-ADEQUACY GATE")
    print("="*92)
    print(f"sites={len(sites)}; context={a.context_days} d; blind starts={a.n_starts}")
    print(f"synthetic sigma_logQ={SYNTH_SIGMA:.4f} is REFERENCE ONLY")

    dummy=np.ones(32,float)
    rescue.simulate_from_theta(dummy,dummy*10,dummy,
                               np.array([10,.5,.005,1.1,.1,1.1,10,1.]))

    for ii,site in enumerate(sites,1):
        print(f"\n{site} ({ii}/{len(sites)})")
        try:
            pf=realio.find_forcing(a.forcing_dir,site)
            pq,dc,qc=realio.find_discharge(a.discharge_dir,site)
            f=realio.load_forcing(pf); q=realio.load_q(pq,dc,qc)
            d,good=realio.merged(q,f); rr=realio.runs(d,good)
            w=realio.central_window(rr,a.context_days)
            if w is None:
                raise RuntimeError(f"no complete {a.context_days}-d window")
            i0,i1=w; obs=d.iloc[i0:i1].copy().reset_index(drop=True)
            start=pd.Timestamp(obs.iloc[0]["date"])
            spin=forcing_spinup(f,start,a.warmup_days)
            if len(spin)<a.min_warmup_days:
                raise RuntimeError(f"only {len(spin)} d forcing spin-up")

            pb=build_problem(spin,obs)
            best,fits,near,md,retry=multistart(
                pb["residual"],len(obs),a.n_starts,a.max_nfev,
                a.retry_starts,a.retry_max_nfev,a.seed+1009*ii)
            th,qsim=simulate_best(pb,best["u"])
            dg=diagnostics(obs["Q"].to_numpy(dtype=float),qsim,obs["date"],pb["floor"])

            fr=dict(wokas_id=site,status="ok",start_date=str(pd.Timestamp(obs.iloc[0]["date"]).date()),
                    end_date=str(pd.Timestamp(obs.iloc[-1]["date"]).date()),context_days=len(obs),
                    warmup_days_used=len(spin),n_starts_total=len(fits),
                    retry_used=retry,best_sse_log=best["sse"],
                    n_nearoptimal_joint95=len(near),max_nearoptimal_u_distance=md)
            fr.update({k:v for k,v in dg.items() if not k.startswith("obs_") and not k.startswith("sim_")})
            FIT.append(fr)

            for z in fits:
                r=dict(wokas_id=site,start_index=z["start_index"],retry_round=z["retry_round"],
                       success=z["success"],nfev=z["nfev"],sse_log=z["sse"],
                       delta_deviance=z["delta_deviance"],
                       nearoptimal_joint95=z["delta_deviance"]<=JOINT95)
                for j,p in enumerate(PARAMS):r["u_"+p]=z["u"][j]
                RUN.append(r)

            pr={"wokas_id":site}
            for j,p in enumerate(PARAMS):
                pr["u_"+p]=best["u"][j]; pr[p]=th[j]
            U=np.vstack([z["u"] for z in near])
            for j,p in enumerate(PARAMS):pr["nearoptimal_u_range_"+p]=np.ptp(U[:,j])
            PAR.append(pr)

            DIA.append({"wokas_id":site,**dg})
            WIN.append(dict(wokas_id=site,start_date=str(pd.Timestamp(obs.iloc[0]["date"]).date()),
                            end_date=str(pd.Timestamp(obs.iloc[-1]["date"]).date()),
                            warmup_days_used=len(spin),
                            warmup_start_date=str(pd.Timestamp(spin.iloc[0]["date"]).date()),
                            warmup_end_date=str(pd.Timestamp(spin.iloc[-1]["date"]).date())))
            lr=np.log(np.maximum(qsim,pb["floor"]))-np.log(np.maximum(obs["Q"].to_numpy(dtype=float),pb["floor"]))
            for k in range(len(obs)):
                PRED.append(dict(wokas_id=site,date=str(pd.Timestamp(obs.iloc[k]["date"]).date()),
                                 Q_obs_m3s=float(obs.iloc[k]["Q"]),Q_sim_m3s=float(qsim[k]),
                                 log_residual_sim_minus_obs=float(lr[k])))

            print(f"  logRMSE={dg['rmse_logQ']:.4f} ({dg['rmse_logQ_over_synthetic_sigma']:.1f}x synth); "
                  f"KGE={dg['KGE']:.3f}; logNSE={dg['logNSE']:.3f}; "
                  f"resACF1={dg['residual_acf1']:.3f}; near={len(near)}; maxDist={md:.3f}")

        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            FIT.append(dict(wokas_id=site,status="error",error=f"{type(e).__name__}: {e}"))

        # progressive saves
        pd.DataFrame(FIT).to_csv(out/"real_exact_primary_fits_PARTIAL.csv",index=False)
        pd.DataFrame(RUN).to_csv(out/"real_exact_multistart_runs_PARTIAL.csv",index=False)
        pd.DataFrame(PAR).to_csv(out/"real_parameter_estimates_PARTIAL.csv",index=False)
        pd.DataFrame(DIA).to_csv(out/"real_predictive_diagnostics_PARTIAL.csv",index=False)
        pd.DataFrame(PRED).to_csv(out/"real_primary_predictions_PARTIAL.csv",index=False)
        pd.DataFrame(WIN).to_csv(out/"real_selected_windows_PARTIAL.csv",index=False)

    F=pd.DataFrame(FIT); R=pd.DataFrame(RUN); P=pd.DataFrame(PAR)
    D=pd.DataFrame(DIA); Y=pd.DataFrame(PRED); W=pd.DataFrame(WIN)
    F.to_csv(out/"real_exact_primary_fits.csv",index=False)
    R.to_csv(out/"real_exact_multistart_runs.csv",index=False)
    P.to_csv(out/"real_parameter_estimates.csv",index=False)
    D.to_csv(out/"real_predictive_diagnostics.csv",index=False)
    Y.to_csv(out/"real_primary_predictions.csv",index=False)
    W.to_csv(out/"real_selected_windows.csv",index=False)

    ok=F[F.status=="ok"].copy()
    summary=dict(n_requested_sites=len(sites),n_successful_sites=len(ok),
                 n_failed_sites=len(F)-len(ok),context_days=a.context_days,
                 synthetic_sigma_logQ_reference=SYNTH_SIGMA,
                 real_sigma_not_assumed_equal_to_synthetic=True,
                 no_discharge_interpolation=True,no_APIC=True,no_model_change=True)
    for c in ["rmse_logQ","rmse_logQ_over_synthetic_sigma","NSE","logNSE","KGE",
              "percent_bias","residual_acf1","residual_acf7","residual_acf30",
              "max_nearoptimal_u_distance"]:
        if c in ok:
            x=pd.to_numeric(ok[c],errors="coerce").dropna()
            if len(x):
                summary["median_"+c]=float(x.median())
                summary["p25_"+c]=float(x.quantile(.25))
                summary["p75_"+c]=float(x.quantile(.75))
    if len(ok):
        summary["fraction_retry_used"]=float(ok.retry_used.astype(bool).mean())
        summary["fraction_distinct_nearoptimal_u_gt_0_45"]=float(
            (ok.max_nearoptimal_u_distance>=.45).mean())
    save_json(summary,out/"REAL_EXACT_INVERSION_STAGE1_SUMMARY.json")

    print("\n"+"="*92)
    print("REAL EXACT INVERSION — STAGE 1 SUMMARY")
    print("="*92)
    print(json.dumps(summary,indent=2))
    print("\nDo not interpret real parameters physically yet. Stage 2 (temporal windows + "
          "profile likelihood) is justified only after predictive adequacy is assessed.")


if __name__=="__main__":
    main()
