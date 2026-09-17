#!/usr/bin/env python3
"""Exploratory hardware/latency correlation analysis for final KITTI runs."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "test_results/dissertation_final/kitti"
OUT = DATA / "hardware_ape_analysis"
MODES = {"baseline": ("ORB-SLAM3", "#4C78A8", "o"),
         "humanslam": ("HuMemSLAM", "#F58518", "s")}
METRICS = [
    ("orb_tracking_mean_ms", "Mean ORB tracking latency (ms)"),
    ("cpu_mean_pct", "Mean CPU utilisation (%)"),
    ("load_mean", "Mean 1-minute system load"),
    ("memory_used_mean_gib", "Mean memory used (GiB)"),
    ("temperature_max_c", "Maximum system temperature (°C)"),
    ("gpu_mean_pct", "Mean GPU utilisation (%)"),
    ("gpu_memory_mean_mib", "Mean GPU memory used (MiB)"),
    ("gpu_temperature_max_c", "Maximum GPU temperature (°C)"),
]


def number(row, key):
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return None


def mean(values):
    values=[x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def maximum(values):
    values=[x for x in values if x is not None and np.isfinite(x)]
    return float(np.max(values)) if values else None


def hardware(path):
    with path.open(newline="") as stream: rows=list(csv.DictReader(stream))
    total_mib = None
    before = path.parent / "system_before.json"
    if before.exists():
        data=json.loads(before.read_text()); total_mib=number(data.get("memory_kib",{}),"MemTotal")
        if total_mib is not None: total_mib /= 1024
    available=[number(row,"memory_available_mib") for row in rows]
    used=[total_mib-x for x in available if x is not None] if total_mib else []
    return {
        "cpu_mean_pct":mean([number(r,"cpu_utilisation_percent") for r in rows]),
        "load_mean":mean([number(r,"load_1m") for r in rows]),
        "memory_used_mean_gib":mean(used)/1024 if used else None,
        "temperature_max_c":maximum([number(r,"max_temperature_c") for r in rows]),
        "gpu_mean_pct":mean([number(r,"gpu_utilisation_percent") for r in rows]),
        "gpu_memory_mean_mib":mean([number(r,"gpu_memory_used_mib") for r in rows]),
        "gpu_temperature_max_c":maximum([number(r,"gpu_temperature_c") for r in rows]),
    }


def run_dirs():
    for condition in sorted(DATA.glob("K0[056]-*")):
        base=condition/"formal_10x" if (condition/"formal_10x").is_dir() else condition
        sequence=condition.name[1:3]
        for run in sorted(base.glob("run_*")):
            for mode in MODES:
                yield sequence,condition.name,run.name,mode,run/mode


def records():
    output=[]
    for sequence,condition,run,mode,path in run_dirs():
        required=[path/"run_summary.json",path/"hardware_monitor.csv"]
        if not all(x.exists() for x in required): continue
        summary=json.loads(required[0].read_text()); trajectory=summary.get("trajectory_evaluation",{})
        maps=trajectory.get("maps",{})
        valid=(bool(trajectory.get("global_ape_valid")) and len(maps)==1 and
               float(trajectory.get("tracking_completeness") or 0)>=.99)
        if not valid: continue
        ape=next(iter(maps.values()),{}).get("ape",{}).get("rmse")
        if ape is None: continue
        row={"sequence":sequence,"condition":condition,"run":run,"mode":mode,
             "ape_rmse_m":float(ape),"orb_tracking_mean_ms":
             summary.get("orb_tracking_ms",{}).get("mean")}
        row.update(hardware(required[1])); output.append(row)
    return output


def correlation(rows,xkey,ykey="ape_rmse_m"):
    pairs=[(r.get(xkey),r.get(ykey)) for r in rows
           if r.get(xkey) is not None and r.get(ykey) is not None]
    if len(pairs)<3:return len(pairs),None,None
    x,y=map(np.asarray,zip(*pairs)); result=spearmanr(x,y)
    return len(pairs),float(result.statistic),float(result.pvalue)


def scatter_pack(sequence,rows):
    directory=OUT/sequence; directory.mkdir(parents=True,exist_ok=True)
    conditions=sorted({r["condition"] for r in rows})
    cmap=plt.get_cmap("tab10"); colours={c:cmap(i%10) for i,c in enumerate(conditions)}
    fig,axes=plt.subplots(2,4,figsize=(18,9)); stats=[]
    for ax,(key,label) in zip(axes.flat,METRICS):
        for row in rows:
            x=row.get(key)
            if x is None:continue
            _,_,marker=MODES[row["mode"]]
            ax.scatter(x,row["ape_rmse_m"],c=[colours[row["condition"]]],marker=marker,s=42,alpha=.8)
            if row["ape_rmse_m"]>np.percentile([r["ape_rmse_m"] for r in rows],90):
                ax.annotate(row["run"].replace("run_",""),(x,row["ape_rmse_m"]),fontsize=7)
        n,rho,p=correlation(rows,key); stats.append({"sequence":sequence,"analysis":"unadjusted","metric":key,"n":n,"spearman_rho":rho,"p_value":p})
        suffix=f"ρ={rho:.2f}, p={p:.3f}, n={n}" if rho is not None else f"n={n}"
        ax.set_xlabel(label); ax.set_ylabel("APE RMSE (m)"); ax.set_title(suffix,fontsize=10)
        ax.grid(alpha=.22); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"KITTI {sequence}: exploratory hardware/runtime association with valid global APE")
    fig.tight_layout(); fig.savefig(directory/"01_hardware_vs_ape.png",dpi=190); fig.savefig(directory/"01_hardware_vs_ape.pdf"); plt.close(fig)

    paired=[]
    lookup={(r["condition"],r["run"],r["mode"]):r for r in rows}
    for condition in conditions:
        runs=sorted({r["run"] for r in rows if r["condition"]==condition})
        for run in runs:
            b=lookup.get((condition,run,"baseline")); h=lookup.get((condition,run,"humanslam"))
            if not b or not h:continue
            item={"sequence":sequence,"condition":condition,"run":run,
                  "delta_ape_rmse_m":h["ape_rmse_m"]-b["ape_rmse_m"]}
            for key,_ in METRICS:
                item["delta_"+key]=(h[key]-b[key]) if h.get(key) is not None and b.get(key) is not None else None
            paired.append(item)
    fig,axes=plt.subplots(2,4,figsize=(18,9))
    for ax,(key,label) in zip(axes.flat,METRICS):
        dk="delta_"+key
        for row in paired:
            if row.get(dk) is None:continue
            ax.scatter(row[dk],row["delta_ape_rmse_m"],c=[colours[row["condition"]]],s=45,alpha=.85)
        n,rho,p=correlation(paired,dk,"delta_ape_rmse_m"); stats.append({"sequence":sequence,"analysis":"paired_delta","metric":key,"n":n,"spearman_rho":rho,"p_value":p})
        suffix=f"ρ={rho:.2f}, p={p:.3f}, n={n}" if rho is not None else f"n={n}"
        ax.axhline(0,color="grey",lw=.8);ax.axvline(0,color="grey",lw=.8)
        ax.set_xlabel("HuMemSLAM − baseline: "+label);ax.set_ylabel("ΔAPE RMSE (m)");ax.set_title(suffix,fontsize=10)
        ax.grid(alpha=.22);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"KITTI {sequence}: paired hardware/runtime differences versus paired APE difference")
    fig.tight_layout();fig.savefig(directory/"02_paired_differences_vs_ape.png",dpi=190);fig.savefig(directory/"02_paired_differences_vs_ape.pdf");plt.close(fig)
    return paired,stats


def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=records(); all_pairs=[];all_stats=[]
    for sequence in ("06","00","05"):
        subset=[r for r in rows if r["sequence"]==sequence]
        pairs,stats=scatter_pack(sequence,subset);all_pairs+=pairs;all_stats+=stats
    for name,data in (("valid_run_hardware_ape.csv",rows),("paired_differences.csv",all_pairs),("spearman_summary.csv",all_stats)):
        with (OUT/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(OUT)


if __name__=="__main__":main()
